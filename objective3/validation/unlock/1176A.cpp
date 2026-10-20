#include <bits/stdc++.h>
using namespace std;
int main(){int q;cin>>q;while(q--){long long n;cin>>n;long long ops=0;while(n%2==0){n/=2;ops+=1;}
while(n%3==0){n/=3;ops+=2;}while(n%5==0){n/=5;ops+=3;}cout<<(n==1?ops:-1)<<"\n";}}
