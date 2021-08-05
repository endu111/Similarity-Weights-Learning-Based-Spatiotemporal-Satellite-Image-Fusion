# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 03:05:49 2020

@author: shx
"""

import torch.nn as nn
import numpy as np
import torch
import time



"""
name:window-based two prediction results fusion function
paramaters:
    method: 1 standard deviation all bands; 
            2 standard deviation each band; 
            3 mean all bands; 
            4 mean each band;
    rimgx: t1 high-spatial-resolution image
    rimgy: t2 high-spatial-resolution image
    rdimgx: t2-t3 low-spatial-resolution differential image
    rdimgy: t2-t1 low-spatial-resolution differential image
    
case:
    #read raw MODIS image
    m1=imgread(modis1file)
    m2=imgread(modis2file)
    m3=imgread(modis3file)  
    #read prediction results1(t1-t2) and prediction results3(t3-t2)
    l2_fake1=imgread(fusion12file)
    l2_fake3=imgread(fusion32file)  
    #set parameters
    param={'part_shape':(60,60),
           'window_size':(41,41),
           'method':1,}
    #run
    l2_fake13=twofs.fuse_main(l2_fake1,l2_fake3,m2-m1,m2-m3,param) 
"""
def fuse_main(rimgx,rimgy,rdimgx,rdimgy,
                param={'part_shape':(140,140),
               'window_size':(31,31),'method':1,}):
    
    ##array to tensor
    rimgx=torch.tensor(rimgx ,dtype=torch.float32).unsqueeze(0)
    rimgy=torch.tensor(rimgy ,dtype=torch.float32).unsqueeze(0)
    rdimgx=torch.tensor(rdimgx ,dtype=torch.float32).unsqueeze(0)
    rdimgy=torch.tensor(rdimgy ,dtype=torch.float32).unsqueeze(0)
    
    #get start time
    time_start=time.time()  

    
    #read parameters
    parts_shape=param['part_shape']
    window=param['window_size']
    method=param['method']    

    #work window size
    padrow=window[0]//2
    padcol=window[1]//2 
    
    #padding low-spatial-resolution image with constant. 
    #fusion one high-spatial-resolution image pixel location based on a low-spatial-resolution image window
    constant=0    
    dimgx =torch.nn.functional.pad( rdimgx ,(padrow,padcol,padrow,padcol),'constant', constant)    
    dimgy =torch.nn.functional.pad( rdimgy ,(padrow,padcol,padrow,padcol),'constant', constant)    
    
    
    #high-spatial-resolution image shape
    imageshape=(rimgx.shape[1],rimgx.shape[2],rimgx.shape[3])
    print('high-spatial-resolution image shape:',imageshape)
    
    #high-spatial-resolution image block shape
    row=imageshape[1]//parts_shape[0]+1
    col=imageshape[2]//parts_shape[1]+1
    
    #generate all splitted index array for extracting high-spatial-resolution image block
    row_part=np.array_split( np.arange(imageshape[1]), row , axis = 0) 
    col_part=np.array_split( np.arange(imageshape[2]),  col, axis = 0) 
    print('Split into {} parts,row number: {},col number: {}'.format(len(row_part)*len(row_part),len(row_part),len(row_part)))
    
    
    #run fusion function for every part
    for rnumber,row_index in enumerate(row_part):
        for cnumber,col_index in enumerate(col_part):
            #run for part: (rnumber,cnumber)
            #print('now for part{}'.format((rnumber,cnumber)))
            
            #the index array and shape  for extracting high-spatial-resolution image block
            rawindex=np.meshgrid(row_index,col_index)
            rawindexshape=(col_index.shape[0],row_index.shape[0])
            
            #the index array and shape for extracting low-spatial-resolution image block
            row_pad=np.arange(row_index[0],row_index[len(row_index)-1]+window[0])
            col_pad=np.arange(col_index[0],col_index[len(col_index)-1]+window[1])    
            padindex=np.meshgrid(row_pad,col_pad)
            padindexshape=(col_pad.shape[0],row_pad.shape[0])
            
            #extract block data from raw image and fusion this block
            if cnumber==0:
                rowdata=  fuse(
                    allband_arrayindex([rimgx,rimgy],rawindex,(1,imageshape[0],rawindexshape[0],rawindexshape[1])),
                    allband_arrayindex([dimgx,dimgy],padindex,(1,imageshape[0],padindexshape[0],padindexshape[1])),
                    window,method)
                
            else:
                rowdata=torch.cat( (rowdata,
                                    fuse(
                    allband_arrayindex([rimgx,rimgy],rawindex,(1,imageshape[0],rawindexshape[0],rawindexshape[1])),
                    allband_arrayindex([dimgx,dimgy],padindex,(1,imageshape[0],padindexshape[0],padindexshape[1])),
                    window,method)
                
                                    ),2) 
        ####Splicing each row        
        if rnumber==0:
            l2_fake=rowdata
        else:            
            l2_fake=torch.cat((l2_fake,rowdata),3)
   
    l2_fake=l2_fake.transpose(3,2)
    
    #time cost
    time_end=time.time()    
    print('now over,use time {:.4f}'.format(time_end-time_start))  
    
    #gpu to cpu
    if torch.cuda.is_available():
        return l2_fake[0].detach().cpu().numpy()
    else:
        return l2_fake[0].detach().numpy()

        
     

"""
name:pytorch-based gpu fusion function
paramaters:
    method: 1 standard deviation all bands; 
            2 standard deviation each band; 
            3 mean all bands; 
            4 mean each band;
    imglist: high-spatial-resolution image list
    dimglist: low-spatial-resolution differential image list
    window: work window size
    
"""
def fuse(imglist,dimglist,window,method=1):
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    #read data
    [imgx,imgy]=imglist
    [dimgx,dimgy]=dimglist
    imgx=imgx.to(device) 
    imgy=imgy.to(device) 
    dimgx=dimgx.to(device) 
    dimgy=dimgy.to(device) 
    
    # b
    bandsize=imgx.shape[1]
    # (h,w)
    outshape=imgx.shape[2:4]
    #### h*w
    blocksize=outshape[0]*outshape[1]
    #### 1,b,h*w
    imgx=nn.functional.unfold(imgx,(1,1))
    imgy=nn.functional.unfold(imgy,(1,1))
    #### b,h*w,windowx*windowy
    dimgx=nn.functional.unfold(dimgx,window).view(bandsize,blocksize,-1)
    dimgy=nn.functional.unfold(dimgy,window).view(bandsize,blocksize,-1)
    
    #fusion method
    #2 standard deviation each band; 
    if method==2:
        wx=[]
        wy=[]
        for i in range(dimgx.shape[1]):
            wxin=(1/abs(dimgx[:,i].std()) )/(1/abs(dimgx[:,i].std())+1/abs(dimgy[:,i].std()))
            wyin=(1/abs(dimgy[:,i].std()) )/(1/abs(dimgx[:,i].std())+1/abs(dimgy[:,i].std()))     
            wx.append(wxin)
            wy.append(wyin)
                
        wx=torch.tensor(wx).unsqueeze(1).unsqueeze(1).to(device)
        wy=torch.tensor(wy).unsqueeze(1).unsqueeze(1).to(device)
        
    #3 mean all bands; 
    elif method==3:
        #### b,h*w
        mstdx=dimgx.mean(2)
        mstdy=dimgy.mean(2)
        ####norm 1,h*w
        wx=(1/(abs( mstdx )+0.0001))/(1/(abs( mstdx)+0.0001)+1/(abs(mstdy )+0.0001))
        wy=(1/(abs( mstdy )+0.0001))/(1/(abs( mstdx)+0.0001)+1/(abs(mstdy )+0.0001))  
        
    #4 mean each band;   
    elif method==4:
        wx=[]
        wy=[]
        for i in range(dimgx.shape[1]):
            wxin=(1/abs(dimgx[:,i].mean()) )/(1/abs(dimgx[:,i].mean())+1/abs(dimgy[:,i].mean()))
            wyin=(1/abs(dimgy[:,i].mean()) )/(1/abs(dimgx[:,i].mean())+1/abs(dimgy[:,i].mean()))     
            wx.append(wxin)
            wy.append(wyin)
               
        wx=torch.tensor(wx).unsqueeze(1).unsqueeze(1).to(device)
        wy=torch.tensor(wy).unsqueeze(1).unsqueeze(1).to(device)     
        
    #1 standard deviation all bands;   
    else:
        #### b,h*w
        stdx=dimgx.std(2)
        stdy=dimgy.std(2)
        ####norm 1,h*w
        wx=(1/ stdx )/(1/ stdx+1/stdy)
        wy=(1/stdy )/(1/stdx+1/stdy )         


    ####fusion (1,1,h*w)X(1,b,h*w)=(1,b,h*w)
    fake=wx*imgx+wy*imgy
    fake=nn.functional.fold(fake,outshape,(1,1))
    return fake



"""
name:extract block data from imagelist based on indexarray
paramaters:
    arraylist: [raw image array1,raw image array2,...]
    indexarray(w X h): index array,2-D array can read like array[indexarray]
    rawindexshape: the set output block shape
"""
def allband_arrayindex(arraylist,indexarray,rawindexshape):
    shape=arraylist[0].shape
    datalist=[]
    for array in arraylist:
        newarray=torch.zeros(rawindexshape,dtype=torch.float32)
        for band in range(shape[1]):
            newarray[0,band]=array[0,band][indexarray]
        datalist.append(newarray)
    return  datalist



