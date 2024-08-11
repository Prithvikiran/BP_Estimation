#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy as sc

from scipy.signal import butter, filtfilt, find_peaks, peak_prominences
from scipy import integrate
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler

from matplotlib import pyplot as plt


# In[2]:


import scipy.io
import pandas as pd 

all_data = pd.DataFrame()

for part in range(1, 13):
    path = f'/kaggle/input/BloodPressureDataset/part_{part}.mat'
    data = scipy.io.loadmat(path)
    
    x = data['p']
    df = pd.DataFrame(x).T
    ppg = [df[0][i][0] for i in range(1000)]
    abp = [df[0][i][1] for i in range(1000)]
    ecg = [df[0][i][2] for i in range(1000)]
    
    df['ppg'] = ppg
    df['abp'] = abp
    df['ecg'] = ecg
    
    all_data = pd.concat([all_data, df], ignore_index=True)


# In[3]:


all_data


# In[4]:


aligned_abp = []
for i in range(12000):
    print(i)
    ppg_signal = all_data['ppg'][i]
    abp_signal = all_data['abp'][i]
    
    cross_corr = np.correlate(ppg_signal, abp_signal, mode='full')
    time_lag = np.argmax(cross_corr) - len(ppg_signal) + 1
    aligned_abp_signal = np.roll(abp_signal, -time_lag)

    aligned_abp.append(aligned_abp_signal)


# In[5]:


all_data['aligned_abp'] = aligned_abp


# In[7]:


new_lst = []


# In[8]:


for i in range(12000):
    new_lst.append([all_data['ppg'].iloc[i], all_data['aligned_abp'].iloc[i]])


# In[9]:


len(new_lst)


# In[10]:


all_data['ppg_abp'] = new_lst


# In[11]:


all_data


# In[12]:


def sampling_freq(signal, time=120):   
    return int(len(signal)/time)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_filtering(signal):
    # sampling freq
    fs = 125
    b, a = butter_bandpass(0.5, 8, fs, order=4)
    
    return filtfilt(b, a, signal)
    


# In[19]:


# SAMPLE TO TEST
sample = all_data['ppg_abp'][0]
plt.plot(sample[0][:1000])


# In[20]:


plt.plot(sample[1][:1000])


# In[21]:


import matplotlib.pyplot as plt

# Sample data for PPG and ABP
ppg_data = sample[0][:300]
abp_data = sample[1][:300]
x_data = range(1, len(ppg_data) + 1)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time')
ax1.set_ylabel('PPG', color=color)
ax1.plot(x_data, ppg_data, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('ABP', color=color)
ax2.plot(x_data, abp_data, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.show()


# In[19]:


fil = apply_filtering(sample)


# In[20]:


plt.figure(figsize=(17,6))

plt.plot(fil, label="filtered PPG")
plt.plot(sample[0], c='gray', alpha=0.4, label="raw PPG")
plt.title("Filtered PPG signal", fontsize=14)
plt.xlabel("Samples", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.legend(prop={'size':14})
plt.xlim(0,500)


# In[63]:


def systolic_peaks(signal):
    return find_peaks(signal, distance=40)[0]

def tfn_points(signal):
    return find_peaks(signal*(-1), height=0, distance=40)[0]

def beat_segmentation(signal, abp):
    
    
    systolics = systolic_peaks(signal)
    tfns = tfn_points(signal)
    
    beats, systolic, sbp_lst, dbp_lst = [], [], [], []
    
    for i in range(len(tfns)-1):
        start = tfns[i]
        end = tfns[i+1]
        segment = np.arange(start, end)
        sbp = np.max(abp[segment])
        dbp = np.min(abp[segment])
        l = [f in systolics for f in segment]
        
        if list(map(bool, l)).count(True) == 1: 
            bshape = signal[segment].shape
            normalized_beat = normalize(signal[segment].reshape(1, -1))
            beats.append(normalized_beat.reshape(bshape))
            systolic.append(np.where(l)[0][0])
            sbp_lst.append(sbp)
            dbp_lst.append(dbp)
            
    
    return beats, systolic, sbp_lst, dbp_lst


# In[34]:


fil


# In[35]:


tfn_points = tfn_points(fil)
tfn_points


# In[36]:


new = systolic_peaks(fil)
new


# In[37]:


plt.figure(figsize = (10, 6))
plt.plot(fil)
plt.scatter(tfn_points, fil[tfn_points], c="m", label="tfn")
plt.scatter(new, fil[new], c="g", label="systolic peaks")


# In[383]:


# Visualize detection of points
systolics = systolic_peaks(fil)
tfns = tfn_points(fil)


plt.figure(figsize=(17,6))
plt.plot(fil, c="r", alpha=0.6)
plt.scatter(systolics, fil[systolics], c="g", label="systolic peaks")
plt.scatter(tfns, fil[tfns], c="m", label="tfn")
plt.legend();


# In[81]:


beats, systolics, sbp, dbp = beat_segmentation(fil, all_data['aligned_abp'][1233])


# In[82]:


len(beats)


# In[83]:


len(sbp)


# In[84]:


def dicrotic_notch(beat, systolic):
    
    derviative = np.diff(np.diff(beat[systolic:]))
    point = find_peaks(derviative)[0]
    corrected = 0
    
    if len(point) > 1:
        corrected =  systolic + point[-2]
        
    return corrected

def diastolic_peak(beat, systolic):
   
    derviative = np.diff(np.diff(beat[systolic:]))
    point = find_peaks(derviative*(-1))[0]
    corrected = 0
    
    if len(point) > 1:
        corrected = systolic + point[-2]
        if abs(beat[corrected]) >= abs(1.01*beat[corrected - 1]):
            return corrected
        else: return 0
        
    return corrected


# In[128]:


def peaks_detection(beats, systolics, sbp, dbp):
    
    dicrotics = []
    diastolics = []
    #print(len(beats))
    #print(len(sbp))
    sbp_lst = []
    dbp_lst = []
    
    for b, s, sp, dp in zip(beats, systolics, sbp, dbp):
        tnn = dicrotic_notch(b,s)
        tdn = diastolic_peak(b,s)
        
        dicrotics.append(tnn)
        diastolics.append(tdn)
        sbp_lst.append(sp)
        dbp_lst.append(dp)
    
    #print(len(dicrotics))
    #print(len(sbp_lst))
    result = np.array([beats, systolics, dicrotics, diastolics, sbp_lst, dbp_lst], dtype=object)
    result = result[..., result[2] > 0]
    result = result[..., result[3] > 0]
    #mask = (result[2] > 0) & (result[3] > 0)
    #result = result[:, mask]
    
    # output shape is (4, nb) where nb is number of beats
    return result.T


# In[98]:


beats_features = peaks_detection(beats, systolics, sbp, dbp)


# In[78]:


len(beats)


# In[100]:


# Visualize detected peaks
for beat, systolic, dicrotic, diastolic, sbp, dbp in beats_features:
    plt.figure()
    plt.plot(beat)
    plt.scatter(systolic,beat[systolic], c="g")
    plt.scatter(diastolic,beat[diastolic], c= "r")
    plt.scatter(dicrotic,beat[dicrotic], c= "gray")
    plt.show()


# In[112]:


def heart_rate(signal, fs):
    
    sys = systolic_peaks(signal)
    T = len(signal)/fs
    
    return len(sys)/(T/60)

def reflection_index(beat, systolic, diastolic):
    
    a = beat[systolic] - np.min(beat)
    b = a - (beat[diastolic] - np.min(beat))
    
    return a/b

def valley_to_valley(beat, fs):
    return len(beat)/fs

def peak_to_valley(beat, systolic, fs):
    return len(beat[systolic:])/fs

def systolic_timespan(dicrotic, fs):
    
    return dicrotic/fs

def up_time(systolic, fs):
    
    return systolic/fs
    
def systolic_volume(beat, dicrotic, fs):
    
    return integrate.simps(beat[:dicrotic], dx=1/fs)
    
def diastolic_volume(beat, dicrotic, fs):
    
    return integrate.simps(beat[dicrotic:], dx=1/fs)


# These features are ***heart rate (HR)***, **crest time (CT)**, **diastolic time** or
# **peak-to-valley interval** (PVI), **valley-to-valley interval** (VVI), **peak-to-peak interval** (PPI), and **peak width at notch level (PW)**.

# In[134]:


def extract_features(signal, beats_arr, fs):
    features = pd.DataFrame()
    
    hr_lst = []
    ri_lst = []
    st_lst = []
    ut_lst = []
    sv_lst = []
    dv_lst = []
    vv_lst = []
    pv_lst = []
    #signal_lst = []
    sbp_lst = []
    dbp_lst = []
    for beat, systolic, dicrotic, diastolic, sbp, dbp in beats_arr:
        
        hr = heart_rate(signal, fs)
        ri = reflection_index(beat, systolic, diastolic)
        st = systolic_timespan(dicrotic, fs)
        ut = up_time(systolic, fs)
        sv = systolic_volume(beat, dicrotic, fs)
        dv = diastolic_volume(beat, dicrotic, fs)
        vv  = valley_to_valley(beat, fs)
        pv = peak_to_valley(beat, systolic, fs)
        sbp_value = sbp
        dbp_value = dbp
        #print(sbp_value)
        #print(dbp_value)
        
        hr_lst.append(hr)
        ri_lst.append(ri)
        st_lst.append(st)
        ut_lst.append(ut)
        sv_lst.append(sv)
        dv_lst.append(dv)
        vv_lst.append(vv)
        pv_lst.append(pv)
        #signal_lst.append(beat)
        sbp_lst.append(sbp)
        dbp_lst.append(dbp)
        
    #features['signal'] = signal_lst    
    features['hr'] = hr_lst
    features['ri'] = ri_lst
    features['st'] = st_lst
    features['ut'] = ut_lst
    features['sv'] = sv_lst
    features['dv'] = dv_lst
    features['vv'] = vv_lst
    features['pv'] = pv_lst
    features['sbp'] = sbp_lst
    features['dbp'] = dbp_lst
    #(Tx, 6)
    return features


# In[135]:


dummy = all_data[:5]


# In[136]:


def process_signal(data):    
    fs = 125
    features_list = []  
    
    for index, row in data.iterrows():
        print(index)
        filtered = apply_filtering(row['ppg'])  
        beats, sys, sbp, dbp = beat_segmentation(filtered, row['aligned_abp'])
        peaks = peaks_detection(beats, sys, sbp, dbp)
        features = extract_features(row['ppg'], peaks, fs)  
        features_list.append(features)
        
    processed_features = pd.concat(features_list, ignore_index=True)
    
    return processed_features


# In[181]:


# features = process_signal(all_data[:300])


# In[182]:


features


# In[201]:


def classify_bp(row):
    
    sbp = row['sbp']
    dbp = row['dbp']
    if sbp < 120 and dbp < 80:
        return 0
    elif 120 <= sbp < 130 and dbp < 80:
        return 1
    elif 130 <= sbp < 140 or 80 <= dbp < 90:
        return 2
    elif 140 <= sbp < 180 or 90 <= dbp < 120:
        return 3
    elif sbp >= 180 or dbp >= 120:
        return 4
    else:
        return 5


# In[202]:


features['bp_class'] = features.apply(classify_bp, axis = 1)


# In[203]:


features['bp_class'].value_counts()


# In[192]:


import joblib 


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import pandas as pd
import os

def get_model_size(model):
    model_filename = "temp_model.joblib"
    joblib.dump(model, model_filename)
    size_kb = os.path.getsize(model_filename) / 1024
    os.remove(model_filename)
    return size_kb

def evaluate_models(data):
    X = data.drop(['sbp', 'dbp', 'bp_class'], axis=1)
    y = data['sbp']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

    results_dict = {'Model': [], 'Mean Absolute Error': [], 'Root Mean Square': [], 'Model Size (KB)': []}

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = (mean_squared_error(y_test, y_pred))**0.5
        size_kb = get_model_size(model)

        results_dict['Model'].append(model_name)
        results_dict['Mean Absolute Error'].append(mae)
        results_dict['Root Mean Square'].append(rmse)
        results_dict['Model Size (KB)'].append(size_kb)

    results_df = pd.DataFrame(results_dict)

    return results_df

models = {
    'RandomForestRegressor': RandomForestRegressor(),
    'LinearRegression': LinearRegression(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'SVR': SVR(),
}

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

models.update({
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'AdaBoostRegressor': AdaBoostRegressor(),
    'MLPRegressor': MLPRegressor(),
    'XGBRegressor': XGBRegressor(),
})


# In[190]:


evaluate_models(features)


# In[191]:


features


# In[214]:


features_test


# In[215]:


features_test['bp_class'] = features_test.apply(classify_bp, axis = 1)


# In[216]:


features_test['bp_class'].value_counts()


# In[166]:


evaluate_models(features)


# In[1]:


# features_test = process_signal(all_data[300:310])


# In[194]:


features


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


import os
import matplotlib.pyplot as plt

def change_to_working_directory():
    working_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(working_dir+"/..")

def plot_loss(loss_curve):
    plt.plot(loss_curve)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('training curve')


# In[ ]:





# In[2]:


import ast
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def scale(lst):
    #lst = ast.literal_eval(lst)
    #lst = np.array(lst).reshape(-1, 1)
    lst = lst.reshape(-1, 1)
    scaled_lst = scaler.fit_transform(lst)
    scaled_lst = scaled_lst.flatten()
    return scaled_lst.tolist()


# In[3]:


all_data['ppg'] = all_data['ppg'].apply(scale)


# In[4]:


stat_features = pd.read_csv('/kaggle/input/kauchee-features-stat/kauchee-stat-corrected-all.csv')


# In[5]:


all_data


# In[6]:


stat_features


# In[7]:


def segment(lst):
    return lst[:1000]


# In[8]:


all_data['ppg'] = all_data['ppg'].apply(segment)


# In[9]:


len(all_data['ppg'][5555])


# In[10]:


all_data['sbp'] = stat_features['sbp']
all_data['SKEWNESS'] = stat_features['SKEWNESS']
all_data['MAXIMUM'] = stat_features['MAXIMUM']
all_data['MINIMUM'] = stat_features['MINIMUM']
all_data['ABSOLUTE_MEAN'] = stat_features['ABSOLUTE MEAN']
all_data['KURTOSIS'] = stat_features['KURTOSIS']




# 	MAXIMUM 	MINIMUM 	ABSOLUTE_MEAN 	KURTOSIS


# In[11]:


all_data['dbp'] = stat_features['dbp']


# In[12]:


all_data


# In[13]:


all_data = all_data.drop([0, 'abp', 'ecg'], axis = 1)


# In[14]:


all_data


# In[15]:


all_data = all_data[(all_data['sbp'] < 180) & (all_data['dbp'] < 90)]


# In[16]:


all_data


# In[23]:


ppg_input = all_data['ppg']


# In[18]:


ppg_input


# In[24]:


demo_input = all_data[['SKEWNESS', 'MAXIMUM', 'MINIMUM', 'ABSOLUTE_MEAN', 'KURTOSIS']]


# In[25]:


sbp_output, dbp_output = all_data['sbp'], all_data['dbp']


# In[37]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Concatenate, Dense

ppg_input = Input(shape=(1000, 1), name='ppg_input')
ppg_gru1 = GRU(64, activation='relu', return_sequences=True)(ppg_input)
ppg_gru2 = GRU(64, activation='relu', return_sequences=True)(ppg_gru1)
ppg_gru3 = GRU(64, activation='relu')(ppg_gru2)

demo_input = Input(shape=(5, 1), name='demo_input')
demo_gru1 = GRU(32, activation='relu', return_sequences=True)(demo_input)
demo_gru2 = GRU(32, activation='relu', return_sequences=True)(demo_gru1)
demo_gru3 = GRU(32, activation='relu')(demo_gru2)

concatenated_output = Concatenate()([ppg_gru3, demo_gru3])

dense1 = Dense(64, activation='relu')(concatenated_output)
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(16, activation='relu')(dense2)
dense4 = Dense(8, activation='relu')(dense3)

sbp_output = Dense(1, activation='linear', name='sbp_output')(dense4)
dbp_output = Dense(1, activation='linear', name='dbp_output')(dense4)

model = Model(inputs=[ppg_input, demo_input], outputs=[sbp_output, dbp_output])

optimizer = tf.keras.optimizers.Adamax(learning_rate=0.005)
model.compile(optimizer=optimizer,
              loss={'sbp_output': 'mse', 'dbp_output': 'mse'},
              metrics={'sbp_output': 'mae', 'dbp_output': 'mae'})

model.summary()


# In[26]:


from sklearn.model_selection import train_test_split

X_ppg_train, X_ppg_test, X_demo_train, X_demo_test, y_sbp_train, y_sbp_test, y_dbp_train, y_dbp_test = train_test_split(
    ppg_input, demo_input, sbp_output, dbp_output , test_size=0.2, random_state=42
)


# In[27]:


X_ppg_train_new = pd.DataFrame()


# In[28]:


X_ppg_train_new['ppg'] = X_ppg_train


# In[29]:


X_ppg_train_new['ppg']


# In[30]:


import numpy as np


# In[32]:


sequences = X_ppg_train_new['ppg'].tolist()
X = np.array(sequences)

X = X.reshape(-1, 1000, 1)


# In[33]:


X


# In[34]:


x_train = np.asarray(X_demo_train).astype(np.float32)


# In[35]:


x_train


# In[ ]:


model.fit([X, x_train], [y_sbp_train, y_dbp_train], batch_size = 128, epochs = 30, validation_split=0.2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


fs = 125
nyq = 0.5 * fs  
lpf_cutoff = 20
order = 4


# In[5]:


from scipy.signal import filtfilt,butter

def butter_lowpass_filter(data, cutoff, fs, order):
    y = data
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, y,padlen=6)
    return y


# In[6]:


all_data['ppg'] = all_data['ppg'].apply(butter_lowpass_filter, args = (lpf_cutoff, fs, order))


# In[7]:


from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

def WhittakerSmooth(x,w,lambda_,differences=1):

    X=np.matrix(x)
    m=X.size
    E=eye(m,format='csc')
    for i in range(differences):
        E=E[1:]-E[:-1] 
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*E.T*E))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)


# In[8]:


def airPLS(x, lambda_=5, porder=1, itermax=15):
    
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return x - z
lambdaa = 5
porder = 1
itermax = 15


# In[9]:


import numpy as np


# In[ ]:


all_data['ppg_corrected'] = all_data['ppg'].apply(airPLS, args = (lambdaa, porder, itermax))


# In[ ]:


all_data['ppg_corrected'] = all_data['ppg_corrected'].apply(scale)


# In[ ]:


def skewness(signal):
    N=len(signal)
    std=np.std(signal)
    mean=np.mean(signal)
    n=0
    for i in range(N):
        n+=((signal[i]-mean)/std)**3
    S=(1/N)*n
    return S


# In[ ]:


list1=[]
for i in range(len(df['ppg_corrected'])):
    list1.append(skewness(df['ppg_corrected'][i][:1000]))
    
list1


# In[ ]:




list3=[]
for i in range(len(df['ppg_corrected'])):
    list3.append(max(df['ppg_corrected'][i][:1000]))
list3


# In[ ]:


list4=[]
for i in range(len(df['ppg_corrected'])):
    list4.append(min(df['ppg_corrected'][i][:1000]))
list4


# In[ ]:


def mean_abs(signal):
    N=len(signal)
    x=np.mean(signal)
    a=0
    for i in range(N):
        a+=abs(signal[i]-x)
    M=(1/N)*a
    return M


# In[ ]:




list5=[]
for i in range(len(df['ppg_corrected'])):
    list5.append(mean_abs(df['ppg_corrected'][i][:1000]))
    
list5


# In[ ]:




def kurtosis(signal):
    N=len(signal)
    std=np.std(signal)
    mean=np.mean(signal)
    n=0
    for i in range(N):
        n+=((signal[i]-mean)/std)**4
    S=(1/N)*n
    return S


# In[ ]:




list6=[]
for i in range(len(df['ppg_corrected'])):
    list6.append(kurtosis(df['ppg_corrected'][i][:1000]))
    
list6


# In[ ]:




