//Lets create a history !
#include <bits/stdc++.h>
#define ll long long
#define MOD 1000000007
#define maxx ((ll)1e6+5)
#define fast ios::sync_with_stdio(0); cin.tie(0); cout.tie(0);
#define test ll t; cin>>t; while(t--)
#define int long long
#define INF 1000000000000000005
#define N 1000005

ll fpow(ll x,ll y,ll p=MOD){x=x%p;ll sum=1;while(y){if(y&1)sum=sum*x;sum%=p;y=y>>1;x=x*x;x%=p;}return sum;}
ll fact[1000007]={0};
void facto() {fact[0]=1;fact[1]=1;for(int i=2;i<100007;i++)fact[i]=(fact[i-1]*i)%MOD;}
ll ncr(ll n,ll r) {ll res=1; res=fact[n]; res=(res*(fpow(fact[r],MOD-2)))%MOD; res=(res*(fpow(fact[n-r],MOD-2)))%MOD; return res;}
ll npr(ll n,ll r) {ll res=1; res=fact[n]; res=(res*(fpow(fact[n-r],MOD-2)))%MOD; return res;}
ll modmul(ll x, ll y){return (x*y)%MOD;}
ll modadd(ll x, ll y){return (x+y)%MOD;}
ll modsub(ll x, ll y){return (x-y+MOD)%MOD;}
ll xx[4]={1,0,-1,0};
ll yy[4]={0,1,0,-1};
using namespace std;
   int f[N];

signed main(){
    int n,m;
    cin>>n>>m;
    if(n==1 && m==1){
        cout<<"NO"<<endl;
    }
    else if (n==1){
        cout<<"YES"<<endl;
        cout<<m<<endl;
        cout<<1<<endl;
    }
    else if (m >=2*n ){
        cout<<"YES"<<endl;
        int sum=0;
        for(int i=0;i<n-1;i++){
            cout<<"2"<<" ";
            sum+=2;
        }
        cout<<m-sum<<endl;
        cout<<"1"<<endl;
    }
    else{
        cout<<"NO"<<endl;
    }


}