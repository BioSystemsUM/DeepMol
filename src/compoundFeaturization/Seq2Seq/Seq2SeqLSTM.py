
class Seq2SeqLSTM():
    def __init__(self, batch_size=64, epochs=10, latent_dim=256,lstm_dim=64,lr=0.001, Prints=False):
        #epsilon: fuzzy factor.
        self.batch_size       = batch_size          #maximal number of texts or text pairs in the single mini-batch (positive integer).
        self.epochs           = epochs              #maximal number of training epochs (positive integer).
        self.latent_dim       = latent_dim          #number of units in the LSTM layer (positive integer).
        self.lstm_dim         = lstm_dim
        self.lr               = lr                  #learning rate (positive float)
        self.Prints           = Prints
        self.unroll           = False


        df = Seq2Seq_lstm_helpers.get_data_From_GitHub(None,None)
        df = Seq2Seq_lstm_helpers.cut_Smile(Seq2Seq_lstm_helpers,df,15,79,self.Prints)
        
        smiles_train, smiles_test = train_test_split(df["smiles"], random_state=42)
        Seq2Seq_lstm_helpers.Dummy_Creation(self,df,smiles_train, smiles_test,self.Prints)
        
        self.input_shape = self.X_train.shape[1:]
        self.output_dim = self.Y_train.shape[-1]

        neck_outputs = Seq2SeqLSTM.Encode_Build(self)
        self.model = Seq2SeqLSTM.lstmDense(self,neck_outputs)
        self.model = Seq2SeqLSTM.model_fit(self,self.model)
        self.sample_model = Seq2SeqLSTM.Encoder(self,'softmax')


    def Encode_Build(self):
        self.encoder_inputs = Input(shape=self.input_shape)
        encoder             = LSTM(self.lstm_dim, return_state=True,unroll=self.unroll)
        encoder_outputs, state_h, state_c = encoder(self.encoder_inputs)
        states = Concatenate(axis=-1)([state_h, state_c])
        neck = Dense(self.latent_dim, activation="relu")
        neck_outputs = neck(states)
        return neck_outputs
        
    def lstmDense (self,neck_outputs):
        #Import Keras objects
        self.decode_h        = Dense(self.lstm_dim, activation="relu")
        self.decode_c        = Dense(self.lstm_dim, activation="relu")
        state_h_decoded = self.decode_h(neck_outputs)
        state_c_decoded = self.decode_c(neck_outputs)
        encoder_states  = [state_h_decoded, state_c_decoded]
        decoder_inputs  = Input(shape=self.input_shape)
        decoder_lstm    = LSTM(self.lstm_dim,return_sequences=True,unroll=self.unroll)
        decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense   = Dense(self.output_dim, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        #Define the model, that inputs the training vector for two places, and predicts one character ahead of the input
        model = Model([self.encoder_inputs, decoder_inputs], decoder_outputs)
        if self.Prints is True:
            print(model.summary())
        return model

    def model_fit(self,model):
        h = History()
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=10, min_lr=0.000001, verbose=1, min_delta=1e-5)
        es = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 5, verbose = True, mode ='auto')
        opt=Adam(learning_rate=self.lr) #Default 0.001 origem 0.005
        model.compile(optimizer=opt, loss='categorical_crossentropy')
        model.fit(
            [self.X_train,self.X_train],self.Y_train, epochs = self.epochs, 
            batch_size=self.batch_size, shuffle=True, callbacks=[h, rlr, es], 
            validation_data=([self.X_test,self.X_test],self.Y_test))
        if self.Prints is True:
            plt.plot(h.history["loss"], label="Loss")
            plt.plot(h.history["val_loss"], label="Val_Loss")
            plt.yscale("log")
            plt.legend()
            Seq2SeqLSTM.checkFailure(self,self.model,100)
        return model

    def checkFailure(self,model,numrang=100):
        fail = 0
        for i in range(numrang):
            v = model.predict([self.X_test[i:i+1], self.X_test[i:i+1]]) #Can't be done as output not necessarely 1
            idxs = np.argmax(v, axis=2)
            pred=  "".join([self.int_to_char[h] for h in idxs[0]])[:-1]
            idxs2 = np.argmax(self.X_test[i:i+1], axis=2)
            true =  "".join([self.int_to_char[k] for k in idxs2[0]])[1:]
            if true != pred:
                if self.Prints is True:
                    print(true, pred)
                fail=fail+1
        print("Failed"+str(fail))

    def Encoder(self,activation_Fun):
        latent_input = Input(shape=(self.latent_dim,))
        #reuse_layers
        state_h_decoded_2 =  self.decode_h(latent_input)
        state_c_decoded_2 =  self.decode_c(latent_input)
        latent_to_states_model = Model(latent_input, [state_h_decoded_2, state_c_decoded_2])
        latent_to_states_model.save("Blog_simple_lat2state.h5")
        #Last one is special, we need to change it to stateful, and change the input shape
        inf_decoder_inputs = Input(batch_shape=(1, 1, self.input_shape[1]))
        inf_decoder_lstm = LSTM(self.lstm_dim,return_sequences=True,unroll=self.unroll,stateful=True)
        inf_decoder_outputs = inf_decoder_lstm(inf_decoder_inputs)
        inf_decoder_dense = Dense(self.output_dim, activation=activation_Fun)
        inf_decoder_outputs = inf_decoder_dense(inf_decoder_outputs)
        sample_model = Model(inf_decoder_inputs, inf_decoder_outputs)
        #Transfer Weights
        for i in range(1,3):
            sample_model.layers[i].set_weights(self.model.layers[i+6].get_weights())
        sample_model.save("Blog_simple_samplemodel.h5")
        sample_model.summary()
        return sample_model


        