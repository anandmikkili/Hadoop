from pysparkling import *
hc = H2OContext.getOrCreate(sc)
import time
import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
input1=h2o.import_file("/home/admin/RECO/cluster8.csv",sep=",")
input_df = input1.as_data_frame(use_pandas=True)
input=input_df[['R_MSISDN','R_PRODUCT_ID','M1_VOICE_REVENUE','M1_TOTAL_VOICE_DURATION','M1_DATA_REVENUE','M1_TOTAL_DATA_VOLUME','M1_SMS_REVENUE','M1_SMS_COUNT','M1_ACTIVE_DAYS','M1_ENGAGEMENT_INDEX','RATING']].copy()
input.head(5)
input=h2o.H2OFrame(input,column_names=["R_MSISDN","R_PRODUCT_ID","M1_VOICE_REVENUE","M1_TOTAL_VOICE_DURATION","M1_DATA_REVENUE","M1_TOTAL_DATA_VOLUME","M1_SMS_REVENUE","M1_SMS_COUNT","M1_ACTIVE_DAYS","M1_ENGAGEMENT_INDEX","RATING"],column_types=["numeric", "factor", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric"])
train,test=input.split_frame(ratios=[.65])
y="R_PRODUCT_ID"
x=train.names[2:11]
dl_1 = H2ODeepLearningEstimator(nfolds = 3,
             fold_assignment = "Modulo",
             keep_cross_validation_predictions = True,
             overwrite_with_best_model=False,
             epochs=10,hidden=[5,5],                      
             score_validation_samples=10000, 
             score_duty_cycle=0.025,         
             adaptive_rate=False,               
             rate=0.01, 
             activation = "Tanh",
             rate_annealing=2e-6,            
             momentum_start=0.2,            
             momentum_stable=0.4, 
             momentum_ramp=1e7, 
             l1=1e-5,                        
             l2=1e-5,  
             seed = 1)
start = time.time()
dl_1.train(x, y, train)
done = time.time()
elapsed = done - start
print(elapsed)
pred = dl_1.predict(test)
pred.head()
performance = dl_1.model_performance(prostate)
performance.show()
model_path = h2o.save_model(model = dl_1, path = "/tmp/mymodel", force = True)
print model_path
h2o.shutdown()