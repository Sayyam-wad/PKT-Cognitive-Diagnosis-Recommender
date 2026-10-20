"""Convert raw C++ source into the Codeforces ASTNN token-tree format (prototype).

Reverse-engineered from the decoded `astnn_*` files of the pkt-attn dataset:

* Parser: libclang 13 (12 behaves the same; 10/11 and >=14 differ), run WITHOUT
  system/standard headers, after expanding #define macros (pcpp).
  Library calls (cin/cout/scanf, vector methods...) are therefore unresolved and
  disappear, `string`/`vector` variables recover as `int`, and conditions that
  involve unresolved operands become empty - exactly what the dataset shows.
* Node shapes mimic pycparser-style ASTNN trees built from clang cursors, e.g.
  FUNCTION_DECL(NAME(main), TYPE_KIND(TYPE(int)), PARM_DECL..., COMPOUND_STMT)
  VAR_DECL(TYPE_KIND(TYPE(long long)), DECLARATOR(NAME(t), <init>))
  CALL_EXPR(NAME(f), ARGUMENTS(...)), BINARY_OPERATOR_<op>, UNARY_OPERATOR_POST_++
* Statement blocks follow ASTNN's get_blocks: FUNCTION_DECL / IF / FOR / WHILE /
  DO nodes are roots, followed by their non-compound children; recursion
  collects nested blocks (including those inside struct methods).
* Tokens are mapped with the dataset's node_w2v_128 vocabulary; unknown tokens
  map to the reserved index len(vocab) (= 32972), as in the dataset.

Exact parity verified on real source/submission pairs recovered from the public
open-r1/codeforces-submissions dataset (see OPTION_C_FEASIBILITY.md). Rules fitted on
those pairs: single-statement bodies wrapped in COMPOUND_STMT (not else-if), functional
casts carry TYPE_KIND, macros expanded before token-based operator detection.
"""

from __future__ import annotations

import os

import clang.cindex as ci
from clang.cindex import CursorKind as K, TypeKind as T

BLOCK_KINDS = {"FUNCTION_DECL", "IF_STMT", "FOR_STMT", "WHILE_STMT", "DO_STMT"}
LITERALS = {K.INTEGER_LITERAL, K.FLOATING_LITERAL, K.CHARACTER_LITERAL, K.STRING_LITERAL, K.CXX_BOOL_LITERAL_EXPR}
SKIP_REFS = {K.TYPE_REF, K.TEMPLATE_REF, K.NAMESPACE_REF}
ARGS = ["-x", "c++", "-std=c++14", "-nostdinc", "-nostdinc++"]


def configure(library_file=None):
    if library_file and not ci.Config.loaded:
        ci.Config.set_library_file(library_file)


CONTROL = {K.IF_STMT, K.FOR_STMT, K.WHILE_STMT, K.DO_STMT}


def _header_end(c):
    """Offset that separates a control statement's header from its body.

    if/for/while: just past the ')' that closes the header; do: the start of the final 'while'.
    """
    toks = list(c.get_tokens())
    if c.kind == K.DO_STMT:
        whiles = [t for t in toks if t.spelling == "while"]
        return whiles[-1].extent.start.offset if whiles else 1 << 60
    depth = 0
    for t in toks:
        if t.spelling == "(":
            depth += 1
        elif t.spelling == ")":
            depth -= 1
            if depth == 0:
                return t.extent.end.offset
    return -1


def _expand_macros(code):
    """Expand #define macros (the dataset's AST was built after preprocessing, but operator
    spellings are read from tokens, which are wrong inside macro expansions)."""
    if "#define" not in code:
        return code
    try:
        import io
        import pcpp

        class _P(pcpp.Preprocessor):
            def on_include_not_found(self, is_malformed, is_system_include, curdir, includepath):
                raise pcpp.OutputDirective(pcpp.Action.IgnoreAndPassThrough)

            def on_error(self, file, line, msg):
                pass

        pp = _P()
        pp.line_directive = None
        pp.parse(code, "submission.cpp")
        out = io.StringIO()
        pp.write(out)
        return out.getvalue()
    except Exception:  # noqa: BLE001 - fall back to the raw source
        return code


def _tokens(c):
    return [t.spelling for t in c.get_tokens()]


def _in_file(c, path):
    return c.location.file is not None and os.path.abspath(c.location.file.name) == path


def type_node(t):
    if t.kind == T.LVALUEREFERENCE:
        return ["LVALUEREFERENCE", type_node(t.get_pointee())]
    if t.kind == T.POINTER:
        return ["POINTER", type_node(t.get_pointee())]
    if t.is_const_qualified() and t.kind not in (T.CONSTANTARRAY, T.INCOMPLETEARRAY, T.VARIABLEARRAY):
        base = t.spelling.replace("const ", "", 1)
        inner = ["TYPE_REF", [base]] if t.kind in (T.TYPEDEF, T.ELABORATED, T.RECORD) else ["TYPE", [base]]
        return ["CONST_QUALIFIED", inner]
    if t.kind in (T.TYPEDEF, T.ELABORATED, T.RECORD, T.UNEXPOSED):
        return ["TYPE_REF", [t.spelling]]
    return ["TYPE", [t.spelling]]


def _operator(c, kids):
    """Operator spelling for BINARY/UNARY/COMPOUND_ASSIGNMENT cursors."""
    toks = list(c.get_tokens())
    if c.kind == K.UNARY_OPERATOR:
        if not toks:
            return "UNARY_OPERATOR_?"
        first, last = toks[0].spelling, toks[-1].spelling
        if last in ("++", "--") and first not in ("++", "--"):
            return f"UNARY_OPERATOR_POST_{last}"
        if first in ("++", "--"):
            return f"UNARY_OPERATOR_PRE_{first}"
        return f"UNARY_OPERATOR_{first}"
    # binary: first operator-like token after the left operand's extent
    prefix = "COMPOUND_ASSIGNMENT_OPERATOR_" if c.kind == K.COMPOUND_ASSIGNMENT_OPERATOR else "BINARY_OPERATOR_"
    left = kids[0] if kids else None
    if left is not None:
        end = left.extent.end.offset
        for tok in toks:
            if tok.extent.start.offset >= end and tok.kind == ci.TokenKind.PUNCTUATION or \
                    tok.extent.start.offset >= end and tok.spelling in ("and", "or", "xor", "bitand", "bitor"):
                return prefix + tok.spelling
    for tok in toks:  # fallback: any operator punctuation inside the expression
        if tok.kind == ci.TokenKind.PUNCTUATION and tok.spelling not in "()[]{};,.":
            return prefix + tok.spelling
    return prefix + "?"


def convert(c, path):
    """Return a list of converted nodes (UNEXPOSED cursors are lifted)."""
    kind = c.kind
    kids = [k for k in c.get_children()]
    if kind == K.UNEXPOSED_EXPR:
        out = []
        for k in kids:
            out += convert(k, path)
        return out
    if kind in SKIP_REFS:
        return [["TYPE_REF", [c.spelling]]] if kind == K.TYPE_REF else []
    if kind == K.OVERLOADED_DECL_REF:
        return []
    if kind in (K.FUNCTION_DECL,):
        if not c.is_definition():
            return []
        node = ["FUNCTION_DECL", ["NAME", [c.spelling]], ["TYPE_KIND", type_node(c.type.get_result())]]
        for k in kids:
            if k.kind == K.PARM_DECL or k.kind == K.COMPOUND_STMT:
                node += convert(k, path)
        return [node]
    if kind in (K.VAR_DECL, K.PARM_DECL, K.FIELD_DECL):
        t = c.type
        exprs = []
        for k in kids:
            if k.kind not in SKIP_REFS:
                exprs += convert(k, path)
        if t.kind in (T.CONSTANTARRAY, T.VARIABLEARRAY, T.INCOMPLETEARRAY, T.DEPENDENTSIZEDARRAY):
            dims, et = 0, t
            while et.kind in (T.CONSTANTARRAY, T.VARIABLEARRAY, T.INCOMPLETEARRAY, T.DEPENDENTSIZEDARRAY):
                dims += 1
                et = et.element_type
            sizes, init = exprs[:dims], exprs[dims:]
            tk = ["TYPE_KIND", ["TYPE_ARRAY", ["ARRAY_SIZES", *sizes], type_node(et)]]
        else:
            tk, init = ["TYPE_KIND", type_node(t)], exprs
        return [[kind.name, tk, ["DECLARATOR", ["NAME", [c.spelling]], *init]]]
    if kind == K.DECL_REF_EXPR:
        return [["DECL_REF_EXPR", [c.spelling]]] if c.spelling else []
    if kind == K.MEMBER_REF_EXPR:
        node = ["MEMBER_REF_EXPR", [c.spelling]]
        for k in kids:
            node += convert(k, path)
        return [node]
    if kind in LITERALS:
        toks = _tokens(c)
        return [[kind.name, [toks[0] if toks else c.spelling]]]
    if kind in (K.BINARY_OPERATOR, K.UNARY_OPERATOR, K.COMPOUND_ASSIGNMENT_OPERATOR):
        node = [_operator(c, kids)]
        for k in kids:
            node += convert(k, path)
        return [node]
    if kind == K.CALL_EXPR:
        args = []
        for a in c.get_arguments():
            args += convert(a, path)
        node = ["CALL_EXPR", ["NAME", [c.spelling or "FUNCTION_CALL"]]]
        if args:
            node.append(["ARGUMENTS", *args])
        return [node]
    if kind in CONTROL:
        # bodies that are single statements are wrapped in COMPOUND_STMT (else-if chains are not)
        hdr = _header_end(c)
        node, n_body = [kind.name], 0
        for k in kids:
            conv = convert(k, path)
            start = k.extent.start.offset
            is_body = start < hdr if kind == K.DO_STMT else start >= hdr
            if is_body:
                n_body += 1
                is_else_if = kind == K.IF_STMT and n_body == 2 and k.kind == K.IF_STMT
                if k.kind != K.COMPOUND_STMT and not is_else_if:
                    conv = [["COMPOUND_STMT", *conv]]
            node += conv
        return [node]
    if kind == K.CXX_FUNCTIONAL_CAST_EXPR:
        node = ["CXX_FUNCTIONAL_CAST_EXPR", ["TYPE_KIND", type_node(c.type)]]
        for k in kids:
            if k.kind not in SKIP_REFS:
                node += convert(k, path)
        return [node]
    if kind == K.CSTYLE_CAST_EXPR:
        node = ["CSTYLE_CAST_EXPR", ["TYPE_KIND", type_node(c.type)]]
        for k in kids:
            node += convert(k, path)
        return [node]
    node = [kind.name]
    for k in kids:
        node += convert(k, path)
    return [node]


def get_blocks(node, out):
    name = node[0]
    children = [ch for ch in node[1:] if isinstance(ch, list) and isinstance(ch[0], str) and len(ch) > 1 or
                (isinstance(ch, list) and ch and ch[0] in BLOCK_KINDS | {"COMPOUND_STMT"})]
    if name in BLOCK_KINDS:
        out.append(node)
        for ch in node[1:]:
            if ch[0] not in BLOCK_KINDS and ch[0] != "COMPOUND_STMT":
                out.append(ch)
            get_blocks(ch, out)
    else:
        for ch in node[1:]:
            get_blocks(ch, out)


def source_to_blocks(code, filename="submission.cpp"):
    """Parse source text and return the ASTNN statement-block list (token strings)."""
    path = os.path.abspath(filename)
    code = _expand_macros(code.replace("\xa0", " ").replace("\r\n", "\n"))
    tu = ci.Index.create().parse(path, args=ARGS, unsaved_files=[(path, code)],
                                 options=ci.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES * 0)
    blocks = []
    for c in tu.cursor.get_children():
        if not _in_file(c, path):
            continue
        for node in convert(c, path):
            get_blocks(node, blocks)
    return blocks


def to_ids(node, key_to_index, unk):
    """Map a token tree to the dataset's integer format [id, child, child...]."""
    tok = node[0]
    out = [key_to_index.get(tok, unk)]
    for ch in node[1:]:
        out.append(to_ids(ch, key_to_index, unk))
    return out


def pretty(node, depth=0, lines=None):
    lines = [] if lines is None else lines
    lines.append("  " * depth + str(node[0]))
    for ch in node[1:]:
        pretty(ch, depth + 1, lines)
    return lines
