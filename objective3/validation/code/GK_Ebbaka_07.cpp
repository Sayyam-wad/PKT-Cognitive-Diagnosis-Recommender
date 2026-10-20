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

  int T;
  cin >> T;

  while (T--) {
    string t;
    cin >> t;

    bool isNice = true;
    for (int i = 0; i < t.size()-1; i++) {
      if (t[i + 1] != t[i]) {
        isNice = false;
      }
    }

    if (isNice) {
      cout << t << endl;
      continue;
    }

    string answer;
    answer.push_back(t[0]);
    for (int i = 1; i < t.size(); i++) {
      if (answer.back() == t[i]) {
        answer.push_back((t[i] == '1') ? '0' : '1');
      }
      answer.push_back(t[i]);
    }

    cout << answer << endl;
  }

  return 0;
}