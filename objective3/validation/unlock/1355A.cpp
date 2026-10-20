#include <bits/stdc++.h>
using namespace std;
int main(){int t;cin>>t;while(t--){long long a,k;cin>>a>>k;for(long long i=1;i<k;i++){long long x=a;int mn=9,mx=0;
while(x){int d=x%10;mn=min(mn,d);mx=max(mx,d);x/=10;}if(mn==0)break;a+=1LL*mn*mx;}cout<<a<<"\n";}}
