class yhat_poetry(YhatModel):
    REQUIREMENTS = reqs
    @preprocess(in_type=dict, out_type=dict)
    def execute(self, data):
        print content
        #Sentiment
        sentiment_features = sentiment(data,inquirer)
        ABS,EnlTot,Female,Male,Object,Polit,Race,Relig,St,WlbPhycs,WlbPsyc,PosNeg = sentiment_features
        #NGrams
        ngram_features = ngram(data,coca)
        unigramFreq,bigramFreq,trigramFreq,misspeltWord, sentence_count,logWordCount,punctFreq = ngram_features
        #Sound
        sound_features = sound(data)
        perfectRhymeFreq,slantRhymeFreq,alliterFreq = sound_features
        #Output - Just few features
        features =pd.DataFrame([perfectRhymeFreq,slantRhymeFreq,Polit,Race,Relig,PosNeg,unigramFreq,bigramFreq,trigramFreq,misspeltWord, sentence_count,logWordCount,punctFreq]).transpose()
        features.columns= ['perfectRhymeFreq','slantRhymeFreq','Polit','Race','Relig','PosNeg','unigramFreq','bigramFreq','trigramFreq','misspeltWord', 'sentence_count','wordCount','punctFreq']       
        #Subset of Percentiles Data
        benchmarks = percentiles[features.columns]
        #Calculate Percentiles
        output =[]
        for i,feature in features.iterrows():
            output.append({k: int(sp.stats.percentileofscore(benchmarks[k],v)) for k,v in feature.iteritems()})
        return output

    #Deploy the class
    yh = Yhat('dzorlu@gmail.com',keys,'http://cloud.yhathq.com/')
    yh.deploy("yhat_poetry", yhat_poetry, globals())