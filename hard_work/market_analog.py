"""Causal nearest-trajectory prediction from observed market hours."""
import numpy as np


def market_features(xm):
    x=np.asarray(xm,dtype=float)
    if x.ndim==3:
        x=x[:,0]
    elif x.ndim==2 and x.shape[0]==24:
        x=x[0:1]
    month=x[:,4]
    return np.column_stack([x[:,8]/.04,x[:,9]/.04,x[:,10]/.04,
                            x[:,11]/.04,x[:,12]/.04,x[:,13]/.04,
                            (x[:,10]-x[:,8])/.015,(x[:,13]-x[:,11])/.015,
                            x[:,20]/.04,x[:,21]/.04,x[:,18]/.06,x[:,19]/.06,
                            np.sin(month*2*np.pi/12)*.7,np.cos(month*2*np.pi/12)*.7,
                            (x[:,2]>=5)*.5])


def build_library(features, exclude=None):
    """Training labels are allowed only if the whole24h trajectory is outside exclude."""
    n=len(features['load'])
    origins=np.arange(n)
    exclude=np.zeros(n,dtype=bool) if exclude is None else np.asarray(exclude,dtype=bool)
    usable=~exclude.copy()
    for h in range(24):
        idx=np.minimum(origins+h,n-1)
        usable &= (origins+h<n)&~exclude[idx]&(features['block_id']==features['block_id'][idx])
    selected=np.flatnonzero(usable)
    x=market_features(features['Xm'])
    med=np.nanmedian(x[selected],axis=0)
    x=np.where(np.isfinite(x),x,med)
    delta_p=features['price'][selected[:,None]+np.arange(24)]-features['price'][selected+2,None]
    delta_c=features['carbon'][selected[:,None]+np.arange(24)]-features['carbon'][selected+2,None]
    return dict(x=x[selected],median=med,hour=features['Xm'][selected,0,5].astype(int),
                delta_price=delta_p,delta_co2=delta_c,
                origins=selected,version=np.asarray(1))


class MarketAnalog:
    def __init__(self,path,k=4):
        with np.load(path,allow_pickle=False) as z:
            self.library={key:z[key] for key in z.files}
        self.k=int(k)
        self.groups={hour:np.flatnonzero(self.library['hour']==hour) for hour in range(24)}

    def predict(self,xm):
        """Input market feature24x44 matrix; output future24 price/CO2 arrays."""
        xm=np.asarray(xm,dtype=float)
        row=xm[0]
        x=market_features(xm)[0]
        lib=self.library
        x=np.where(np.isfinite(x),x,lib['median'])
        pool=self.groups[int(row[5])]
        if not len(pool):
            raise ValueError('No market analogs for origin hour')
        d=np.sum((lib['x'][pool]-x)**2,axis=1)
        k=min(self.k,len(pool))
        picked=np.argpartition(d,k-1)[:k]
        neighbors=pool[picked]
        weights=1/(.2+d[picked])
        weights/=weights.sum()
        forecasts=[]
        for key,start in [('delta_price',8),('delta_co2',11)]:
            known=row[start:start+3]
            finite=np.flatnonzero(np.isfinite(known))
            if not len(finite):
                raise ValueError('Current market observation is absent')
            j=int(finite[-1])
            delta=lib[key][neighbors]
            forecast=known[j]+np.sum(weights[:,None]*(delta-delta[:,j,None]),axis=0)
            forecast[finite]=known[finite]
            if key=='delta_co2':
                forecast=np.maximum(forecast,0.)
            forecasts.append(forecast)
        return forecasts
