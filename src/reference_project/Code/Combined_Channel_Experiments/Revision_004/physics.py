"""P838 coefficients and two explicitly approximate hourly rain-only paths."""
import io
import zipfile
import numpy as np
from common import require

# ITU-R P.838-3 Tables1–4, visually checked against official PDF.
COEFFICIENTS={
 'kh':([-5.33980,-.35351,-.23789,-.94158],[-.10008,1.26970,.86036,.64552],[1.13098,.45400,.15354,.16817],-.18961,.71147),
 'kv':([-3.80595,-3.44965,-.39902,.50167],[.56934,-.22911,.73042,1.07319],[.81061,.51059,.11899,.27195],-.16398,.63297),
 'ah':([-.14318,.29591,.32177,-5.37610,16.1721],[1.82442,.77564,.63773,-.96230,-3.29980],[-.55187,.19822,.13164,1.47828,3.43990],.67849,-1.95537),
 'av':([-.07771,.56727,-.20238,-48.2991,48.5833],[2.33840,.95545,1.14520,.791669,.791459],[-.76284,.54039,.26809,.116226,.116479],-.053739,.83433)}
POLARIZATIONS={'horizontal':0.,'vertical':90.,'circular':45.}
HEIGHTS={'minus050':-.5,'nominal':0.,'plus050':.5}

def base_coefficients(frequency):
    require(np.isfinite(frequency) and 1<=frequency<=1000,'P838 frequency domain GHz1..1000')
    x=np.log10(frequency);result={}
    for name,(a,b,c,m,d) in COEFFICIENTS.items():
        value=np.sum(np.asarray(a)*np.exp(-((x-np.asarray(b))/np.asarray(c))**2))+m*x+d
        result[name]=float(10**value if name.startswith('k') else value)
    return result

def coefficients(frequency,elevation,polarization):
    require(np.isfinite(elevation) and 0<=elevation<=90,'Elevation degrees0..90')
    require(polarization in POLARIZATIONS,'Unknown polarization')
    v=base_coefficients(frequency)
    factor=np.cos(np.deg2rad(elevation))**2*np.cos(np.deg2rad(2*POLARIZATIONS[polarization]))
    k=(v['kh']+v['kv']+(v['kh']-v['kv'])*factor)/2
    alpha=(v['kh']*v['ah']+v['kv']*v['av']+(v['kh']*v['ah']-v['kv']*v['av'])*factor)/(2*k)
    return float(k),float(alpha)

def height_grid(path):
    with zipfile.ZipFile(path) as z:
        require(z.testzip() is None,'P839 ZIP CRC')
        lat,lon,h0=(np.loadtxt(io.BytesIO(z.read(n))) for n in ('Lat.txt','Lon.txt','h0.txt'))
    require(lat.shape==lon.shape==h0.shape==(121,241),'P839 grid shape')
    require(np.array_equal(lat,np.broadcast_to(np.arange(90,-90.1,-1.5)[:,None],lat.shape)),'P839 latitude grid')
    require(np.array_equal(lon,np.broadcast_to(np.arange(0,360.1,1.5)[None,:],lon.shape)),'P839 longitude grid')
    require(np.isfinite(h0).all() and (h0>=0).all() and np.array_equal(h0[:,0],h0[:,-1]),'P839 finite/cyclic map')
    return lat,lon,h0

def climate_height(h0_grid,coordinate_grids,latitude,longitude):
    require(-90<=latitude<=90 and np.isfinite(longitude),'Station coordinate domain')
    x=longitude%360/1.5;y=(90-latitude)/1.5
    j=int(np.floor(x));i=min(int(np.floor(y)),119)
    wx=x-j;wy=y-i
    corners=[(i,j,(1-wy)*(1-wx)),(i,j+1,(1-wy)*wx),
             (i+1,j,wy*(1-wx)),(i+1,j+1,wy*wx)]
    rows=[{'i':ii,'j':jj,'latitude_deg':float(coordinate_grids[0][ii,jj]),'longitude_deg':float(coordinate_grids[1][ii,jj]),'h0_km_amsl':float(h0_grid[ii,jj]),'weight':float(w)} for ii,jj,w in corners]
    h0=sum(r['h0_km_amsl']*r['weight'] for r in rows)
    return {'h0_km_amsl':h0,'hR_km_amsl':h0+.36,'corners':rows}

def attenuation(rain,frequency,elevation,latitude,hs_km,hr_km,polarization):
    require(np.isfinite([elevation,latitude,hs_km,hr_km]).all(),'Finite geometry required')
    require(5<=elevation<=90 and -90<=latitude<=90,'Only declared theta>=5 branch supported')
    require(hr_km>=0,'Negative AMSL rain height outside declared domain')
    rain=np.asarray(rain,dtype=float);valid=np.isfinite(rain)&(rain>=0)
    k,alpha=coefficients(frequency,elevation,polarization)
    geom=np.full(rain.shape,np.nan);proxy=geom.copy();gamma=geom.copy()
    theta=np.deg2rad(elevation);depth=max(hr_km-hs_km,0.)
    gamma[valid]=k*rain[valid]**alpha
    slant=depth/np.sin(theta)
    geom[valid]=gamma[valid]*slant
    if depth==0:proxy[valid]=0.
    else:
        g=gamma[valid];horizontal=slant*np.cos(theta)
        reduction=1/(1+.78*np.sqrt(horizontal*g/frequency)-.38*(1-np.exp(-2*horizontal)))
        zeta=np.rad2deg(np.arctan2(depth,horizontal*reduction))
        # At theta90 the else branch is used, avoiding division by zero.
        lr=np.full_like(g,slant);use=zeta>elevation
        lr[use]=horizontal*reduction[use]/np.cos(theta)
        chi=max(36-abs(latitude),0.)
        vertical=1/(1+np.sqrt(np.sin(theta))*(31*(1-np.exp(-elevation/(1+chi)))*np.sqrt(lr*g)/frequency**2-.45))
        proxy[valid]=g*lr*vertical
    require(np.isfinite(geom[valid]).all() and np.isfinite(proxy[valid]).all() and (geom[valid]>=0).all() and (proxy[valid]>=0).all(),'Physical finite/nonnegative gate')
    return {'geom':geom,'p618_hourly_proxy':proxy,'gamma_db_per_km':gamma,'slant_km':float(slant),'k':k,'alpha':alpha}
