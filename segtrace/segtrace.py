import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from pomegranate  import *
from uuid import uuid4
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class Segtrace:
    ts_col = 'converted_to_timestamp'
    ts_until_next_col = 'timestamp_until_next_event'
    cut_col = 'cut_col'
    time_cluster_col = '__time_labels__'
    topic_col = '__topic_labels__'

    def __init__(self, df, case_col='', activity_col='', datetime_col=''):
        self.df = df
        self.case_col = case_col
        self.activity_col = activity_col
        self.datetime_col = datetime_col
        self.check_columns()
        self.data_preparation()
        self.analysis = self.data_analysis()

    def check_columns(self):
        #if self.cut_col in self.df.columns:
        #    raise ValueError('{} should not already exist as a column'.format(self.cut_col))
        #if self.ts_col in self.df.columns:
        #    raise ValueError('{} should not already exist as a column'.format(self.ts_col))
        #if self.ts_until_next_col in self.df.columns:
        #    raise ValueError('{} should not already exist as a column'.format(self.ts_until_next_col))
        if self.case_col not in self.df.columns:
            raise ValueError('{} is not in the dataframe'.format(self.case_col))
        if self.activity_col not in self.df.columns:
            raise ValueError('{} is not in the dataframe'.format(self.activity_col))
        if self.datetime_col not in self.df.columns:
            raise ValueError('{} is not in the dataframe'.format(self.datetime_col))
        if self.df[self.datetime_col].dtypes != 'datetime64[ns]':
            raise ValueError('{} should be a datetime. Convert it first to date time using pd.to_datetime().'.format(self.datetime_col))

    def data_preparation(self):
        # Convert the datetime to integer
        self.df[self.ts_col] = self.df[self.datetime_col].astype('int64') // 10 ** 6

        # Convert all the other columns to string
        self.df[self.activity_col] = self.df[self.activity_col].astype(str)
        self.df[self.case_col] = self.df[self.case_col].astype(str)

        # Calculate the distance between events
        #   to make sure events belonging to distinct case_id are distant (will not end in the same cluster)
        #   we add a big interval; we use the (timestamp.max() - timestamp.min() + 1)
        self.df[self.ts_until_next_col] = self.df.groupby(self.case_col)[self.ts_col].shift(-1) - self.df[self.ts_col]
        self.df.loc[self.df[self.ts_until_next_col].isna(), self.ts_until_next_col] = \
            self.df[self.ts_until_next_col].max() - self.df[self.ts_until_next_col].min() + 1

    def data_analysis(self):
        '''
        Returns a dataframe useful to do some analysis
        E.g.,
                       s_until_next  count  n_traces  avg_length_trace
            0           0.0           5714     44286          1.129025
            1           1.0             64     44222          1.130659
            2           2.0             35     44187          1.131555
            3           3.0              6     44181          1.131708
            4           4.0              2     44179          1.131759

        Looking at the dataframe, we know that if we insert a cut if the
        time is above 1seconds, we would cut in 44222 pieces and an avg length of 1.130659
        :return: dataframe
        '''
        analysis = self.df[self.ts_until_next_col].value_counts().to_frame(name='count').\
            sort_index().reset_index().rename({'index':'s_until_next'}, axis=1)
        analysis['n_traces'] = self.df.shape[0]-analysis['count'].cumsum()
        analysis['avg_length_trace'] = analysis['count'].sum()/analysis['n_traces']
        return analysis

    def equal_sized_time_interval(self, k, min_avg_length=None, max_avg_length=None):
        '''
        Return k potential time cuts that will be equal sized.
        For instance, it might returns [1, 5, 43, 72] there are
        if there are 100 cuts at 1 (100 times there are more than 1 seconds between events)
        there will be 200 cuts at 5 (200 time there are more than 5 seconds between events)
        and so on...
        :param min_avg_length: We can limit the search space by restricting the min avg length of the trace.
        :param max_avg_length: We can limit the search space by restricting the max avg length of the trace.
        :return: a vector of cuts
        '''
        if min_avg_length is None:
            min_avg_length = 1
        if max_avg_length is None:
            max_avg_length = self.analysis['avg_length_trace'].max()

        min_avg_length = max(1, min_avg_length)
        max_avg_length = min(self.analysis['avg_length_trace'].max(), max_avg_length)

        time_gaps = self.analysis.loc[
            (self.analysis['avg_length_trace']<=max_avg_length) &
            (self.analysis['avg_length_trace']>=min_avg_length)
            ,'s_until_next'].sort_values(ascending=False).values

        time_gaps = [x.right for x in pd.qcut(time_gaps, q=k, precision=1, duplicates='drop').categories]

        return time_gaps

    def cut(self, time_cut):
        return (self.df[self.ts_until_next_col]>=time_cut).shift(1).fillna(0).astype(int).cumsum()


    def directly_follow_feature(self, time_cut, cut=True):
        '''
        Given a time_cut, it extract the directly follow features
        :param time_cut: max time between event. If above => cut the traces
        :param col_cluster: Column that will store the resulting cluster
        :return: None
        '''
        if cut:
            self.df[self.cut_col] = self.cut(time_cut)
        else:
            self.df[self.cut_col] = self.df[time_cut]

        self.df['DF'] = self.df[self.activity_col]+'=>'+self.df.groupby(self.cut_col)[self.activity_col].shift(-1).fillna('end')
        #count = self.df['DF'].value_counts()
        #all_index = self.df.index
        #data = self.df.loc[self.df['DF'].isin(count.nlargest(999999).index),:].pivot_table(index=self.cut_col, columns='DF', values=self.activity_col, aggfunc='count').reindex(index=all_index)>0
        data = self.df.pivot_table(index=self.cut_col, columns='DF', values=self.activity_col, aggfunc='count')>0

        self.df.drop('DF', axis=1, inplace=True)
        svd = TruncatedSVD(n_components=min(10, data.shape[1]-1), n_iter=3)
        data = pd.DataFrame(svd.fit_transform(data), index=data.index)
        return data

    def ngram_features(self, time_cut, cut=True):
        if cut:
            self.df[self.cut_col] = self.cut(time_cut)
        else:
            self.df[self.cut_col] = self.df[time_cut]
        seq = self.df.groupby(self.cut_col)[self.activity_col].agg(list)
        cv = CountVectorizer(ngram_range=(1,1), tokenizer=lambda doc: doc, binary=True, lowercase=False)
        data = cv.fit_transform(seq)
        svd = TruncatedSVD(n_components=min(10, data.shape[1]-1), n_iter=10)
        data = pd.DataFrame(svd.fit_transform(data), index=seq.index)

        return data



    def cluster(self, features, k):
        '''
        Works in three steps:
            1) Cut according to time_cut
            2) Cluster using the directly follow as features with k
            3) Store the resulting cluster for each event in the column col_cluster
        :param time_cut: max time between event. If above => cut the traces
        :param k:  number of clusters to discover
        :param col_cluster: Column that will store the resulting cluster
        :return: None
        '''

        #data = normalize(data, norm='l2')
        #model = GeneralMixtureModel.from_samples(BernoulliDistribution, n_components=k, X=features.astype(int).values, max_iterations=10^6)
        #labels = model.predict(features)
        clusterer = MiniBatchKMeans(n_clusters=k)


        clusterer.labels_ = np.zeros(features.shape[0])
        while np.unique(clusterer.labels_).shape[0] != k:
            clusterer.fit(features)


        return clusterer.labels_

    def build_hmm_model(self, time_cluster_col):

        self.df[self.time_cluster_col] = time_cluster_col

        # Data Preparation
        self.df['previous_cluster'] = self.df.groupby(self.case_col)[self.time_cluster_col].shift(1).fillna('start')
        self.df['first_trace'] = self.df[self.case_col] != self.df[self.case_col].shift(1)
        self.df['last_trace'] = self.df[self.case_col] != self.df[self.case_col].shift(-1)

        # Distribution of activities per cluster
        pivot = self.df.pivot_table(index=self.time_cluster_col, columns=self.activity_col, values=self.case_col, aggfunc='count').fillna(0)
        cluster_index = pivot.index.values  # Making sure the cluster are always in the same order
        distributions = [DiscreteDistribution(x) for x in pivot.div(pivot.sum(axis=1), axis=0).to_dict(orient='index').values()]

        # Trans_mat
        pivot = self.df.loc[self.df['last_trace']==False,:].pivot_table(index=self.time_cluster_col, columns='previous_cluster', values=self.case_col, aggfunc='count').reindex(columns=cluster_index, index=cluster_index).fillna(0)
        trans_mat = pivot.div(pivot.sum(axis=1), axis=0).reindex(index=cluster_index, columns=cluster_index)
        #print (trans_mat.to_string())

        # Starts
        starts = self.df[self.df['first_trace']==True].groupby(self.time_cluster_col)[self.case_col].count().reindex(cluster_index).fillna(0)
        starts = (starts/starts.sum()).transpose().values

        # Ends
        #print (self.df[self.df['last_trace']==True].head(1000).to_string())
        ends = self.df[self.df['last_trace']==True].groupby(self.time_cluster_col)[self.case_col].count().reindex(cluster_index).fillna(0)
        ends = (ends/ends.sum()).transpose().values

        self.df.drop(self.time_cluster_col, axis=1, inplace=True)

        hmm = HMModel()
        hmm.from_segtrace(trans_mat, distributions, starts, ends)

        return hmm

    def sum_log_probability(self, model):
        x = [model.hmmModel.log_probability(x) for x in self.df.groupby(self.case_col)[self.activity_col].agg(list).tolist()]
        return sum(x)

    def get_topic(self, model):
        self.df[self.topic_col] = None
        for i, x in self.df.groupby(self.case_col)[self.activity_col].agg(list).to_dict().items():
            self.df.loc[self.df[self.case_col]==i,self.topic_col] = model.hmmModel.predict(x)
        return self.df[self.topic_col]



class HMModel():

    def __init__(self):
        self.hmmModel = None
        self.time_cluster_col = None
        self.topic_col = None
        self.name = uuid4()
        self.path = 'models/{}'.format(self.name)

    def from_segtrace(self, matrix, distributions, starts, ends):
        self.hmmModel = HiddenMarkovModel.from_matrix(matrix, distributions, starts, ends)

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        with open('{}/model.json'.format(self.path), 'w') as f:
            f.write(self.hmmModel.to_json())

    def load(self, uid):
        self.name = uid
        self.path = 'models/{}'.format(self.name)
        with open('{}/model.json'.format(self.path), 'r') as f:
            self.hmmModel = HiddenMarkovModel.from_json(f.read())


