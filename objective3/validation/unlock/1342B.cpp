#include <bits/stdc++.h>
using namespace std;
int main(){int t;cin>>t;while(t--){string s;cin>>s;bool same=count(s.begin(),s.end(),s[0])==(int)s.size();
if(same){cout<<s<<"\n";}else{string r;for(size_t i=0;i<s.size();i++)r+="01";cout<<r<<"\n";}}}
