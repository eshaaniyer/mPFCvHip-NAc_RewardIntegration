import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sp
import scipy.signal as sg
from pathlib import Path
import pyinform as inf
import math

def dFF(raw1,con1):
    try:
        reg=np.polyfit(np.array(con1),np.array(raw1),1)
        a = reg[0]
        b = reg[1]
        controlFit = np.dot(a,con1) + b
        dF=raw1 - controlFit
        normDat = 100*np.divide(dF,controlFit) #this gives deltaF/F
        normDat = sp.zscore(normDat)
        normDat=pd.DataFrame(data=normDat)
        return normDat
    except:
        return pd.DataFrame(raw1)

def LP_TimeLocked_Traces(Paradigm):
    #phase:PRL, One Lever, No Lever
    j=0

    PFC_array = np.array([])
    vHIP_array = np.array([])
    rewards = np.array([])
    lp = np.array([])
    rt = np.array([])
    ID = np.array([])
    time_array = np.array([])
    index_array = np.array([])
    dayarray = np.array([])
    leftleverprob = np.array([])

    mutual_info = np.array([])
    cond_pv = np.array([])
    cond_vp = np.array([])

    mutual_info_pc = np.array([])
    cond_pv_pc = np.array([])
    cond_vp_pc = np.array([])
    mutual_info_ei = np.array([])
    cond_pv_ei = np.array([])
    cond_vp_ei = np.array([])

    #CREATE CSVs
    PRL_Ref=pd.read_csv("/home/bagotlab/eshaan.i/PRL/FP_PRL_Nov_2021/PRL_Ref.csv")
    for index, row in PRL_Ref.iterrows():
        Animal = str(row["Animal"])
        if Paradigm == 'PRL':
            startday = row['PRL Day 1']
            endday = row['PRL Day 6']
        if Paradigm == 'PRL 4-6':
            startday = row['PRL Day 4']
            endday = row['PRL Day 6']
        if Paradigm == 'One_Lever':
            startday = row['One Lever Day 1']
            endday = row['One Lever Day 3']
        if Paradigm == 'No_Lever':
            startday = row['No Lever Day 1']
            endday = row['No Lever Day 3']
        if Paradigm == 'Final_PRL':
            startday = row['PRL Day 7']
            endday = row['PRL Day 9']
        if math.isnan(startday) or math.isnan(endday):
            print(Animal, 'Has not reached this phase yet')
            continue
        startday =int(startday)
        endday = int(endday)+1
        for d in range (startday, endday):
            Day = d
            Med ="/home/bagotlab/eshaan.i/PRL/FP_PRL_Nov_2021/Med/Day " + str(Day)
            if len(glob(Med +'/*'+Animal+'.txt')) == 0:
                continue
            file = glob(Med +'/*'+Animal+'.txt')[0]
            file = open(file,"r")

            i=0
            while i < 34:
                if(i == 4):
                    # Grab the date the data was recorded on
                    date = file.readline()[12:-1]
                    date=date.replace("/", "")
                if(i==5):
                    # Grab the subject's ID
                    subject_name=file.readline()[8:-1]
                    print("Subject: ", subject_name, 'Day: ', Day)
                else:
                    file.readline()     
                i = i + 1
            #print(subject_name)

            with file as f:
                for l in f:
                    if l.startswith('Z:'):
                        break
                PRL_df =pd.read_fwf(f, names=["Ignore", "Lever A Probability", "Left B Probability", "Lever_Pressed", "Rewards_Delivered", "RT"])

            FPData = "/home/bagotlab/eshaan.i/PRL/FP_PRL_Nov_2021/FP/Day "+str(Day)
            Ref = pd.read_csv(('/home/bagotlab/eshaan.i/PRL/FP_PRL_Nov_2021/FP/Day '+str(Day)+'/Ref_6Fiber.csv'))    

            for index, row in Ref.iterrows():
                if row["Animal"] == int(subject_name):
                    Box = row["Box"]
                    ending = row["Ending"]
                    PFCchannel = int(row["PFC"])
                    VHIPchannel = int(row["VHIP"])
                    #print(Sex)
            
            #print(PFCchannel,VHIPchannel)
            TTLFile = FPData + '/TTL_Box'+str(int(Box))+str(int(ending))+'.csv'
            #print(TTLFile)

            FPFile = FPData + '/FIP_PRL_Nov2021_data'+str(int(ending))+'.csv'
            if Path(FPFile).exists():
                FPFile=pd.read_csv(FPFile, delim_whitespace=True,header=None)
            else: 
                print('No Recordings this day')
                continue
            cons=FPFile[1::2].reset_index(drop=True)
            raws=FPFile[::2].reset_index(drop=True)

            if len(raws) > len(cons):
                raws = raws[0:len(cons)]
            else:
                cons = cons[0:len(raws)]        

            try:
                PFC_raw=raws[PFCchannel]
                PFC_truecon=cons[PFCchannel]
                VHIP_raw=raws[VHIPchannel]
                VHIP_truecon=cons[VHIPchannel]
                #print('PFC is ',PFCchannel,'VHIP is ',VHIPchannel)
            except: 
                if VHIPchannel == 999:
                    VHIP_raw=(raws[0]*0)+1
                    VHIP_truecon=(cons[0]*0)+1
                    PFC_raw=raws[PFCchannel]
                    PFC_truecon=cons[PFCchannel]
                    #print("PFC is ", PFCchannel)
                if PFCchannel == 999:
                    PFC_raw=(raws[0]*0)+1
                    PFC_truecon=(cons[0]*0)+1
                    VHIP_raw=raws[VHIPchannel]
                    VHIP_truecon=cons[VHIPchannel]
                    #print("VHIP is ", VHIPchannel)
            TimeVector = np.array(raws[6])
                
            Time = (TimeVector - TimeVector[0])/1000
            PFC = dFF(PFC_raw,PFC_truecon)
            VHIP = dFF(VHIP_raw,VHIP_truecon)
            
            TTLs = pd.read_csv(TTLFile, names=["TimeStamp","T/F"])
            TTLstart = TTLs[TTLs["T/F"] == False].reset_index(drop=True)
            TTLstart.columns = ['LeverOut', 'T/F']
            TTLend = TTLs[TTLs["T/F"] == True].reset_index(drop=True)
            TTLend.columns = ['LeverIn', 'T/F']
            TTL = pd.concat([TTLstart["LeverOut"],TTLend["LeverIn"],PRL_df],axis=1)
            TTL=TTL.dropna()
            for index, row in TTL.iterrows():
                lppfc = pd.DataFrame()
                lepfc = pd.DataFrame()

                LeverIn = (np.abs(TimeVector-(int(row["LeverIn"])))).argmin()

                LP_pfc=PFC.loc[(LeverIn-40):(LeverIn+199)].reset_index(drop=True).to_numpy().flatten()
                LP_vhip=VHIP.loc[(LeverIn-40):(LeverIn+199)].reset_index(drop=True).to_numpy().flatten()

                if PFCchannel != 999 and len(LP_pfc) == 240:
                    PFC_array=np.concatenate((PFC_array, LP_pfc),axis=0)
                else:
                    PFC_array=np.concatenate((PFC_array, np.full(240,np.nan)),axis=0)

                if VHIPchannel != 999 and len(LP_vhip) == 240:
                    vHIP_array=np.concatenate((vHIP_array, LP_vhip),axis=0)
                else:
                    vHIP_array=np.concatenate((vHIP_array, np.full(240,np.nan)),axis=0)

                if PFCchannel != 999 and VHIPchannel != 999 and len(LP_pfc) == 240:
                    mutual_info=np.concatenate((mutual_info, inf.mutual_info(LP_pfc+100,LP_vhip+100,local=True)),axis=0)
                    cond_pv=np.concatenate((cond_pv,inf.conditional_entropy(LP_pfc+100,LP_vhip+100,local=True)),axis=0)
                    cond_vp=np.concatenate((cond_vp,inf.conditional_entropy(LP_vhip+100,LP_pfc+100,local=True)),axis=0)

                    mutual_info_pc = np.concatenate((mutual_info_pc, np.full(240,inf.mutual_info(LP_pfc[40:80]+100,LP_vhip[40:80]+100))),axis=0)
                    cond_pv_pc = np.concatenate((cond_pv_pc, np.full(240,inf.conditional_entropy(LP_pfc[40:80]+100,LP_vhip[40:80]+100))),axis=0)
                    cond_vp_pc = np.concatenate((cond_vp_pc, np.full(240,inf.conditional_entropy(LP_vhip[40:80]+100,LP_pfc[40:80]+100))),axis=0)
                    
                    mutual_info_ei = np.concatenate((mutual_info_ei, np.full(240,inf.mutual_info(LP_pfc[200:240]+100,LP_vhip[200:240]+100))),axis=0)
                    cond_pv_ei = np.concatenate((cond_pv_ei, np.full(240,inf.conditional_entropy(LP_pfc[200:240]+100,LP_vhip[200:240]+100))),axis=0)
                    cond_vp_ei = np.concatenate((cond_vp_ei, np.full(240,inf.conditional_entropy(LP_vhip[200:240]+100,LP_pfc[200:240]+100))),axis=0)

                else:
                    mutual_info=np.concatenate((mutual_info, np.full(240,np.nan)),axis=0)
                    cond_pv=np.concatenate((cond_pv,np.full(240,np.nan)),axis=0)
                    cond_vp=np.concatenate((cond_vp,np.full(240,np.nan)),axis=0)
                    mutual_info_pc=np.concatenate((mutual_info_pc, np.full(240,np.nan)),axis=0)
                    cond_pv_pc=np.concatenate((cond_pv_pc,np.full(240,np.nan)),axis=0)
                    cond_vp_pc=np.concatenate((cond_vp_pc,np.full(240,np.nan)),axis=0)
                    mutual_info_ei=np.concatenate((mutual_info_ei, np.full(240,np.nan)),axis=0)
                    cond_pv_ei=np.concatenate((cond_pv_ei,np.full(240,np.nan)),axis=0)
                    cond_vp_ei=np.concatenate((cond_vp_ei,np.full(240,np.nan)),axis=0)

                rewards = np.concatenate((rewards, np.full(240,int(row["Rewards_Delivered"]))),axis=0)
                lp = np.concatenate((lp, np.full(240,int(row["Lever_Pressed"]))),axis=0)
                rt = np.concatenate((rt, np.full(240,row["RT"])),axis=0)
                ID =np.concatenate((ID, np.full(240,int(Animal))),axis=0)
                time_array = np.append(time_array, np.linspace(-2, 10, 240))
                index_array = np.concatenate((index_array,np.full(240,j)),axis=0)
                dayarray = np.concatenate((dayarray, np.full(240,Day)),axis=0)
                leftleverprob = np.concatenate((leftleverprob, np.full(240,int(row["Lever A Probability"]))),axis=0)
                j=j+1
    name = '/home/bagotlab/eshaan.i/PRL/FP_PRL_Nov_2021/CLEAN/Data/' + Paradigm + '_ZScore_ITI.csv'
    DF_longform=np.concatenate(([index_array],[time_array],[dayarray],[ID],[leftleverprob],[rewards],[lp],[rt], [PFC_array],[vHIP_array],[mutual_info],[cond_pv],[cond_vp],[mutual_info_pc],[cond_pv_pc],[cond_vp_pc],[mutual_info_ei],[cond_pv_ei],[cond_vp_ei]),axis=0).T
    DF_longform=pd.DataFrame(columns=['index','time','Day','ID','LeftLeverProb','rewards','lp','rt','PFC','vHIP','MutualInfo','CE_PFCvHIP','CE_vHIPPFC','MutualInfo_PC','CE_PFCvHIP_PC','CE_vHIPPFC_PC','MutualInfo_EI','CE_PFCvHIP_EI','CE_vHIPPFC_EI'], data=DF_longform)
    DF_longform.to_csv(name)
    return DF_longform

def Licks_TimeLocked_Traces(Paradigm):
    j=0

    PFC_array = np.array([])
    vHIP_array = np.array([])
    rewards = np.array([])
    lp = np.array([])
    rt = np.array([])
    ID = np.array([])
    time_array = np.array([])
    index_array = np.array([])
    dayarray = np.array([])


    #CREATE CSVs
    PRL_Ref=pd.read_csv("/home/bagotlab/eshaan.i/PRL/FP_PRL_Nov_2021/PRL_Ref.csv")
    for index, row in PRL_Ref.iterrows():
        Animal = str(row["Animal"])
        if Paradigm == 'PRL':
            startday = row['PRL Day 1']
            endday = row['PRL Day 6']
        if Paradigm == 'PRL 4-6':
            startday = row['PRL Day 4']
            endday = row['PRL Day 6']
        if Paradigm == 'One Lever':
            startday = row['One Lever Day 1']
            endday = row['One Lever Day 3']
        if Paradigm == 'No Lever':
            startday = row['No Lever Day 1']
            endday = row['No Lever Day 3']
        if Paradigm == 'Final PRL':
            startday = row['PRL Day 7']
            endday = row['PRL Day 9']
        if math.isnan(startday) or math.isnan(endday):
            print(Animal, 'Has not reached this phase yet')
            continue
        startday =int(startday)
        endday = int(endday)+1
        for d in range (startday, endday):
            Day = d
            Med ="/home/bagotlab/eshaan.i/PRL/FP_PRL_Nov_2021/Med/Day " + str(Day)
            if len(glob(Med +'/*'+Animal+'.txt')) == 0:
                continue
            file = glob(Med +'/*'+Animal+'.txt')[0]
            file = open(file,"r")

            i=0
            while i < 34:
                if(i == 4):
                    # Grab the date the data was recorded on
                    date = file.readline()[12:-1]
                    date=date.replace("/", "")
                if(i==5):
                    # Grab the subject's ID
                    subject_name=file.readline()[8:-1]
                    print("Subject: ", subject_name, 'Day: ', Day)
                else:
                    file.readline()     
                i = i + 1
            #print(subject_name)

            i = 0
            h = 0
            file.seek(0)
            with file as f:
                for l in f:
                    if l.startswith('Z:'):
                        break
                    i=i+1
                PRL_df =pd.read_fwf(f, names=["Ignore", "Lever A Probability", "Left B Probability", "Lever_Pressed", "Rewards_Delivered", "RT"])
            
                file.seek(0)
                for line in file:
                    h = h+1 
                    if line.startswith('U:'):
                        Licks_df = pd.read_fwf(file, nrows=(i-h),colspecs=[(7, 19), (20, 32), (33, 45), (46, 58), (59, 71)])
                        break
                    
                #print("Zrow: {}, Urow: {} \n".format(zrow, urow))
                Licks = (Licks_df.to_numpy().flatten()) * 1000
                if Licks.size == 0:
                    print('lmfao no licks???')
                    Licks = np.array([0])

            FPData = "/home/bagotlab/eshaan.i/PRL/FP_PRL_Nov_2021/FP/Day "+str(Day)
            Ref = pd.read_csv(('/home/bagotlab/eshaan.i/PRL/FP_PRL_Nov_2021/FP/Day '+str(Day)+'/Ref_6Fiber.csv'))    

            for index, row in Ref.iterrows():
                if row["Animal"] == int(subject_name):
                    Box = row["Box"]
                    ending = row["Ending"]
                    PFCchannel = int(row["PFC"])
                    VHIPchannel = int(row["VHIP"])
                    #print(Sex)
            
            #print(PFCchannel,VHIPchannel)
            TTLFile = FPData + '/TTL_Box'+str(int(Box))+str(int(ending))+'.csv'
            #print(TTLFile)

            FPFile = FPData + '/FIP_PRL_Nov2021_data'+str(int(ending))+'.csv'

            if Path(FPFile).exists():
                FPFile=pd.read_csv(FPFile, delim_whitespace=True,header=None)
            else: 
                print('No Recordings this day')
                continue
            cons=FPFile[1::2].reset_index(drop=True)
            raws=FPFile[::2].reset_index(drop=True)

            if len(raws) > len(cons):
                raws = raws[0:len(cons)]
            else:
                cons = cons[0:len(raws)]        

            try:
                PFC_raw=raws[PFCchannel]
                PFC_truecon=cons[PFCchannel]
                VHIP_raw=raws[VHIPchannel]
                VHIP_truecon=cons[VHIPchannel]
                #print('PFC is ',PFCchannel,'VHIP is ',VHIPchannel)
            except: 
                if VHIPchannel == 999:
                    VHIP_raw=(raws[0]*0)+1
                    VHIP_truecon=(cons[0]*0)+1
                    PFC_raw=raws[PFCchannel]
                    PFC_truecon=cons[PFCchannel]
                    #print("PFC is ", PFCchannel)
                if PFCchannel == 999:
                    PFC_raw=(raws[0]*0)+1
                    PFC_truecon=(cons[0]*0)+1
                    VHIP_raw=raws[VHIPchannel]
                    VHIP_truecon=cons[VHIPchannel]
                    #print("VHIP is ", VHIPchannel)
            TimeVector = np.array(raws[6])
                
            Time = (TimeVector - TimeVector[0])/1000
            PFC = dFF(PFC_raw,PFC_truecon)
            VHIP = dFF(VHIP_raw,VHIP_truecon)
            
            TTLs = pd.read_csv(TTLFile, names=["TimeStamp","T/F"])
            TTLstart = TTLs[TTLs["T/F"] == False].reset_index(drop=True)
            TTLstart.columns = ['LeverOut', 'T/F']
            TTLend = TTLs[TTLs["T/F"] == True].reset_index(drop=True)
            TTLend.columns = ['LeverIn', 'T/F']
            TTL = pd.concat([TTLstart["LeverOut"],TTLend["LeverIn"],PRL_df],axis=1)
            TTL=TTL.dropna()
            Licks = Licks + TimeVector[0]

            for index, row in TTL.iterrows():

                LeverIn = (np.abs(TimeVector-(int(row["LeverIn"])))).argmin()
                lick_index = np.nanargmin(np.abs(Licks - int(row["LeverIn"])))
                lick = (np.abs(TimeVector - Licks[lick_index])).argmin()

                if ( lick > LeverIn) & ( lick < LeverIn+200):
                    LP_pfc=PFC.loc[(lick-40):(lick+199)].reset_index(drop=True).to_numpy().flatten()
                    LP_vhip=VHIP.loc[(lick-40):(lick+199)].reset_index(drop=True).to_numpy().flatten()
                else:
                    LP_pfc = np.full(240,np.nan)
                    LP_vhip = np.full(240,np.nan)

                if PFCchannel != 999 and len(LP_pfc) == 240:
                    PFC_array=np.concatenate((PFC_array, LP_pfc),axis=0)
                else:
                    PFC_array=np.concatenate((PFC_array, np.full(240,np.nan)),axis=0)

                if VHIPchannel != 999 and len(LP_vhip) == 240:
                    vHIP_array=np.concatenate((vHIP_array, LP_vhip),axis=0)
                else:
                    vHIP_array=np.concatenate((vHIP_array, np.full(240,np.nan)),axis=0)

                rewards = np.concatenate((rewards, np.full(240,int(row["Rewards_Delivered"]))),axis=0)
                lp = np.concatenate((lp, np.full(240,int(row["Lever_Pressed"]))),axis=0)
                rt = np.concatenate((rt, np.full(240,row["RT"])),axis=0)
                ID =np.concatenate((ID, np.full(240,int(Animal))),axis=0)
                time_array = np.append(time_array, np.linspace(-2, 10, 240))
                index_array = np.concatenate((index_array,np.full(240,j)),axis=0)
                dayarray = np.concatenate((dayarray, np.full(240,Day)),axis=0)
                j=j+1

    name = '/home/bagotlab/eshaan.i/PRL/FP_PRL_Nov_2021/CLEAN/Data/' + Paradigm + '_ZScore_Licks.csv'
    DF_longform=np.concatenate(([index_array],[time_array],[dayarray],[ID],[rewards],[lp],[rt], [PFC_array],[vHIP_array]),axis=0).T
    DF_longform=pd.DataFrame(columns=['index','time','Day','ID','rewards','lp','rt','PFC','vHIP'], data=DF_longform)
    DF_longform.to_csv(name)

LP_TimeLocked_Traces('No_Lever')
LP_TimeLocked_Traces('PRL')
LP_TimeLocked_Traces('One_Lever')

#Licks_TimeLocked_Traces('PRL')
#Licks_TimeLocked_Traces('One_Lever')
#Licks_TimeLocked_Traces('No_Lever')