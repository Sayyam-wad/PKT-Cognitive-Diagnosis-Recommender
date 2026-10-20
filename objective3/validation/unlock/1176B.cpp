#include <bits/stdc++.h>
using namespace std;
int main(){int t;cin>>t;while(t--){int n;cin>>n;int c[3]={0,0,0};for(int i=0;i<n;i++){long long x;cin>>x;c[x%3]++;}
int m=min(c[1],c[2]);cout<<c[0]+m+(c[1]-m)/3+(c[2]-m)/3<<"\n";}}
