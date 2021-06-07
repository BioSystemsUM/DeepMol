class Seq2Seq_lstm_helpers():

    def __init__():
        return self

    def vectorize(self,smiles,embed):
        one_hot =  np.zeros((smiles.shape[0], embed , len(self.charset)),dtype=np.int8)
        for i,smile in enumerate(smiles):
            #encode the startchar
            one_hot[i,0,self.char_to_int["!"]] = 1
            #encode the rest of the chars
            for j,c in enumerate(smile):
                one_hot[i,j+1,self.char_to_int[c]] = 1
            #Encode endchar
            one_hot[i,len(smile)+1:,self.char_to_int["E"]] = 1
        #Return two, one for input and the other for output
        return one_hot[:,0:-1,:], one_hot[:,1:,:]
    
    def get_data_From_GitHub(url, df):
        if url is None:
            url ='https://github.com/GLambard/Molecules_Dataset_Collection/raw/master/originals/HIV.csv'
        df = pd.read_csv(io.StringIO(requests.get('https://github.com/GLambard/Molecules_Dataset_Collection/raw/master/originals/HIV.csv').content.decode('utf-8')), index_col = 0)
        df.reset_index(inplace=True)
        return df
    
    def process_smiles_array(smiles_array):
      lengths = list()
      for i in smiles_array:
          lengths.append(len(i))
      return lengths
    
    def Cut_Range_of_Smiles(df,save,plot,smiles_lengths):
        if save is True:
            df['smiles_length'] = smiles_lengths
        if plot is True:
            plt.hist(smiles_lengths, bins=100)
            plt.ylabel('Number of SMILES')
            plt.xlabel('Length of SMILES')
            plt.show()
        return df
    
    def cut_Smile(self,df,Start,End,Prints):
        if Prints is True:
            print(df.shape)
        smiles_lengths = self.process_smiles_array(df['smiles'].values)
        df = self.Cut_Range_of_Smiles(df,True,Prints,smiles_lengths)
        length_range = (Start,End) #Range of the cut

        if Prints is True:
            filtered = filter(lambda x: length_range[0] <= x <= length_range[1], smiles_lengths)
            percentage = len(list(filtered)) / len(df['smiles'].values)
            print('Percentage of instances with SMILES\' length between %s and %s: %s' % (length_range[0], length_range[1], percentage))
        df = df[(df['smiles_length'] >= length_range[0]) & (df['smiles_length'] <= length_range[1])]
        df = df.drop('smiles_length', axis='columns')
        if Prints is True:
            print(df.shape)
        smiles_lengths = self.process_smiles_array(df['smiles'].values)
        df = self.Cut_Range_of_Smiles(df,False,Prints,smiles_lengths)
        return df
        
    def Dummy_Creation(self,data,smiles_train,smiles_test,Prints):
        self.charset = set("".join(list(data.smiles))+"!E")
        self.char_to_int = dict((c,i) for i,c in enumerate(self.charset))
        self.int_to_char = dict((i,c) for i,c in enumerate(self.charset))
        embed = max([len(smile) for smile in data.smiles]) + 5
        self.X_train, self.Y_train = Seq2Seq_lstm_helpers.vectorize(self,smiles_train.values,embed)
        self.X_test,  self.Y_test = Seq2Seq_lstm_helpers.vectorize(self,smiles_test.values,embed)
        if Prints is True:
            print(str(self.charset))
            print(len(self.charset), embed)
            print(smiles_train.iloc[1])
            plt.matshow(self.X_train[1].T)
            print(self.X_train.shape)