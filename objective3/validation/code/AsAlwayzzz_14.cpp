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
    int a,b,c,d;
    cin>>a>>b>>c>>d;

    for(int i=a;i<=b;i++){
        f[i+b]+=1;
        f[i+c+1]-=1;
    }
    for(int i=1;i<N;i++){
        f[i]+=f[i-1];
    }

    int ans=0;
    for(int i=1;i<=b+c;i++){
        if(f[i]>0){
            if(i>d){
                ans+=f[i]*(d-c+1);
            }
            else if(i<=c){
                ans+=0;
            }
            else{
                    if(i==4){
                     //   cout<<f[i]<<" "<<i<<" "<<c;
                    }
                ans+=f[i]*(i-c);
            }
        }

    }
    cout<<ans<<endl;

}