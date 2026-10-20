#include <bits/stdc++.h>
using namespace std;
int main(){int t;cin>>t;while(t--){long long x,y,a,b;cin>>x>>y>>a>>b;long long lo=min(x,y),hi=max(x,y);
cout<<min((x+y)*a, lo*b+(hi-lo)*a)<<"\n";}}
