import numpy as np
import pandas as pd
from glob import glob
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import re
import yaml

class nlp():
    tokenizer = Tokenizer(num_words=1000, split=" ")
    tfidfvectorizer = TfidfVectorizer(analyzer='word')
    clf = GaussianNB()

    def raw_data_import(self,filename):
        return pd.read_csv(filename)


    def create_vocab(self,tokenizer=tokenizer):
        file_list=glob('data\*.csv')
        all_sent=[]
        for i in file_list:
            a=pd.read_csv(i)
            for j in a.columns:
                li=np.array(a[j]).tolist()
                li = list(filter(lambda x: str(x) != "nan", li))
                all_sent.extend(li)
        all_sent=np.array(all_sent)
        b=all_sent.ravel()
        s=tokenizer.fit_on_texts(list(all_sent))
        a=(tokenizer.word_index)
        vocab=list(a.keys())
        return vocab

    def create_tfidf(self,csv_filename,tfidfvectorizer=tfidfvectorizer):
        intents_df = pd.read_csv(csv_filename)
        tfidf_wm = (tfidfvectorizer.fit_transform(intents_df["user_input"]))
        feature_names = [None]*len(tfidfvectorizer.vocabulary_)
        for key in tfidfvectorizer.vocabulary_:
            feature_names[tfidfvectorizer.vocabulary_[key]] = key
        df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = feature_names)
        return df_tfidfvect

    def tokenize_intents(self,intent_column):
        intents_tokens={}
        zero=0
        for i in intent_column.unique(): 
            intents_tokens[i]= zero # embedding intents to serial numbers Ex.{asking name:0, asking number:1}
            zero+=1
        return intents_tokens

    def output_series(self, intents_df,intents_tokens):
        c=intents_df.copy()
        d=c.replace({"intent": intents_tokens})
        int_df_int_tok=d["intent"]
        return int_df_int_tok

    def vectorize(self,input_data,vec=tfidfvectorizer):
        input_data=input_data.lower()
        tfidf_wm_ip = vec.transform([input_data])
        feature_names = [None]*len(vec.vocabulary_)
        for key in vec.vocabulary_:
            feature_names[vec.vocabulary_[key]] = key
        df_tfidfvect_ip = pd.DataFrame(data = tfidf_wm_ip.toarray(),columns = feature_names)
        return(df_tfidfvect_ip)

        
    def invert_dict(self,dictionary):
        return {value:key for key,value in dictionary.items()}
 
    def check_mail(self,email):
        import re
        from email_validator import validate_email, EmailNotValidError
        tokens= email.split()
        a=0
        for i in tokens:
            if "@" in i:
                email=i
                a+=1
        if a==0:
            return None
        try:
            v = validate_email(email)
            email = v["email"] 
            email=email
            return email
        except EmailNotValidError as e:
            return(None)
        
    def identify_name(self,input:str,name_df,name=None):
        input_tokens=input.split()
        for i in input_tokens:
            if len(name_df[(name_df["name"]).str.lower()==(str(i).lower())]) > 0:
                name=i
                break
        if name:
            return name
        else:
            return None

    def identify_number(self,input:str,number=False):
        import re
        pattern_float = r'\d+\.\d+'
        pattern_number = r'\d+'
        if len(re.findall(pattern_float, input))==0:
            if len(re.findall(pattern_number, input))==0:
                return None
            else:
                number=re.findall(pattern_number, input)
        else:
            number=re.findall(pattern_float, input)
        return(number[0])
        
    
    def identify_pan(self,input:str,number=False):
        import re
        pattern_pan = r'\b[A-Z]{3}[ABCFGHLJPTF]{1}[A-Z]{1}[0-9]{4}[A-Z]{1}\b'
        if len(re.findall(pattern_pan, input.upper()))==0:
                return None
        else:
            pan=re.findall(pattern_pan, input.upper())
            return(pan[0])

    def identify_location(self, input:str):
        input_list=input.split()
        df=pd.read_csv("cities.csv", columns=["city_name"])
        for i in input_list:
            if len(df[(df["city_name"]).str.lower()==(str(i).lower())]) > 0:
                name=i
                break
        if name:
            return name
        else:
            return None

    def read_yaml(self,file_name):
        with open(file_name, "r") as file:
            return yaml.safe_load(file)
    
    def naive_bayes_fit(self,x,y,clf=clf):
        return clf.fit(x,y)

    def naive_bayes_predict(self, x_test, clf=clf):
        return clf.predict(x_test)[0]

    def vec_and_predict(self,user_reply,nlp_object):
        reply_vector = nlp_object.vectorize(user_reply)
        reply_intent=nlp_object.naive_bayes_predict(reply_vector)
        return reply_intent
