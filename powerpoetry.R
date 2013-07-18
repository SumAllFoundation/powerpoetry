#Powerpoetry
#Location#############################################
loc = read.csv('location.csv',header=TRUE)

map("state", interior = FALSE)
map("state", boundary = FALSE, col="gray", add = TRUE)

qm<-function(field,dir,col){
  z<- as.integer(cut2(field, m=500))
  topq= as.integer(z==dir)
  lon = loc$longitude[topq!=0]
  lan = loc$latitude[topq!=0]
  points(lon ,lan,cex=0.5,col=col,pch=19)
}
#PosNeg
z<-with(loc,cut2(PosNeg,g=10))
with(loc,points(longitude,latitude,cex=0.5,col=z,pch=19))

#######################################################
#State Level Aggregation
state = read.csv('state.csv',header=TRUE)
#kNN
trainset = state[,c('latitude','longitute')]
cl = state[,c('StateName')]
target = loc[,c('latitude','longitude')]
#Find the best k. Train and Test.
n = dim(train)[1]*0.7
tr<-sample(dim(train)[1],dim(train)[1]*0.7)
test = train[-tr,]
train = train[tr,]
cltest = cl[-tr]
cltrain = cl[tr]
accuracy = matrix(NA,nrow=10)
for (k in 1:10){
  st = knn(train,test,k=k,cl = cltrain)
  accuracy[k] = mean(st==cltest)
}
#Append Scores with State Information
st = knn(trainset,target,k=1,cl=cl)
#Take the median for each feature by state

for (i in 5:27){
  feature_score<-with(loc,tapply(loc[,i],st,mean))
  poetrymap<- cbind(poetrymap,feature_score)
}
poetrymap<-data.frame(poetrymap)

colnames(poetrymap)[2:24]<-colnames(loc)[5:27]


##############################################
#Features
features = read.csv('progress.csv',header=TRUE)
colnames(features)[1]<-'user'
colnames(features)[2]<-'count'
#Get rid of mod
features <- features[features$user!=0,]
#df <- subset(features, select = -c('nid',) )
#z<-with(sf,tapply(Score,as.factor(count),median))
#Sub Feature. Take the poets with >10 poems posted. 
sub<-function(features,n)
{
  uu<-unique(features$user[features$count>=n])
  sf<-data.frame()
  for (i in 1:length(uu)){ 
    temp <- features[features$user == uu[i],]
    temp <- temp[temp$count<=n,]
    sf<- rbind(sf,temp)
  }
  return(sf) 
}

#Generate slopes for each feature
pdf()
for (i in 18
  {
    #Median score of each feature by each poem count
    feature_score<-with(sf,tapply(sf[,i],as.factor(count),mean))
    #Plot
    with(sf,plot(feature_score,col='red',pch=19))
    #Line
    lm1<-lm(as.matrix(feature_score)~as.numeric(1:n))
    abline(lm1)
    #Title with p-value of the slope
    title(paste(colnames(sf[i])," (p-val of slope:", round(anova(lm1)$'Pr(>F)'[1],2),")"))
  }
dev.off()

