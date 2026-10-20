#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <map>
#include<numeric>
#include<time.h>
#include<cstdlib>
#include <bitset>
#include <set>

typedef long long int LLint;
using namespace std;

int main() {

  int t;
  cin >> t;

  while (t--) {
    LLint x, y;
    cin >> x >> y;

    LLint a, b;
    cin >> a >> b;

    if (x == 0 && y == 0) {
      cout << 0 << endl;
      continue;
    }

    if (x == 0) {
      cout << (LLint)abs(y - (LLint)0) * (LLint)a << endl;
      continue;
    }

    if (y == 0) {
      cout << (LLint)abs(x - (LLint)0) * (LLint)a << endl;
      continue;
    }

    if ( ((x > 0) && (y < 0)) || ((x < 0) && (y > 0))) {
      LLint res = (LLint)(abs(x - (LLint)0)) * a + (LLint)(abs(y - (LLint)0)) * a;
      cout << res;
      continue;
    }

    LLint new_x = abs(x);
    LLint new_y = abs(y);

    LLint firstAns = LLint(new_x * a) + LLint(new_y * a);

    LLint secondAns = (max(new_x, new_y) - min(new_x, new_y)) * a + min(new_x, new_y) * b;
    
    cout << min(firstAns, secondAns) << endl;
  }

  return 0;
}