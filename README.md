# recommend_songs
도로교통공단 교통방송 프로그램별 선곡 추천 시스템 개발


## Factorization machine 학습 과정 (tensorflow 기반)
```python
def train_on_batch(model, optimizer, accuracy, inputs, targets):
    with tf.GradientTape() as tape:
        y_pred,w,V = model(inputs)
        # loss = tf.keras.losses.binary_crossentropy(from_logits=False,
        #                                            y_true=targets,
        #                                            y_pred=y_pred)
        loss = tf.keras.losses.MSE(y_true=targets,y_pred=y_pred)

        # print(w)
        # print(V)
####################
        error = tf.keras.losses.MSE(y_true=targets,y_pred=y_pred)
        regularizer_w = tf.nn.l2_loss(w)
        regularizer_v = tf.nn.l2_loss(V)
        loss = (error + 0.001 * regularizer_w + 0.001 * regularizer_v)

#####################

        # lambda_w = tf.constant(0.001, name='lambda_w')
        # lambda_v = tf.constant(0.001, name='lambda_v')
        

        # l2_norm = (tf.multiply(lambda_w, tf.pow(w, 2))+
        #                 tf.multiply(lambda_v, tf.pow(V, 2)))
        # print(l2_norm)

        # l2_norm = (tf.math.scalar_mul(0.001, tf.pow(w, 2))+
        #                 tf.math.scalar_mul(0.001, tf.pow(V, 2)))

        # l2_norm = (tf.reduce_sum(
        #             tf.add(
        #                 tf.multiply(lambda_w, tf.pow(w, 2)),
        #                 tf.multiply(lambda_v, tf.pow(V, 2)))))
        # loss = tf.add(error, l2_norm)

    
    # loss를 모델의 파라미터로 편미분하여 gradients를 구한다.
    grads = tape.gradient(target=loss, sources=model.trainable_variables)

    # apply_gradients()를 통해 processed gradients를 적용한다.
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # accuracy: update할 때마다 정확도는 누적되어 계산된다.
    accuracy.update_state(targets, y_pred)
    
    return loss


# 반복 학습 함수
def train(X_train, X_test, Y_train, Y_test, p, k, epochs, batch=16, learning_rate=0.001, threshold=0.5):
#    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
    # X_train, X_test, Y_train, Y_test = xtrain, xtest, ytrain, ytest

    train_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_train, tf.float32), tf.cast(Y_train, tf.float32))).shuffle(5000).batch(batch)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_test, tf.float32), tf.cast(Y_test, tf.float32))).shuffle(2000).batch(batch)

    print('data transformaiton completed')

    model = FM(p, k)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    accuracy = RootMeanSquaredError()
    loss_history = []

    #checkpoint 생성
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer = optimizer, model = model)
    manager = tf.train.CheckpointManager(ckpt, '/content/drive/MyDrive/temp/fm_check', max_to_keep=3)
    
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
      print("Restored from {}".format(manager.latest_checkpoint))
    else:
      print("Initializing from scratch.")
      
    for i in range(epochs):
      for x, y in train_ds:
          loss = train_on_batch(model, optimizer, accuracy, x, y)
          loss_history.append(loss)
      ckpt.step.assign_add(1)
      if int(ckpt.step) % 1 == 0:
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
      # status.assert_consumed()
      # checkpoint.save(checkpoint_prefix)

      # print(accuracy)

      if i % 1== 0:
          # print("스텝 {:03d}에서 누적 평균 손실: {:.4f} 정확도: {:.4f}".format(i, np.mean(loss_history), accuracy.result().numpy()))
          print("스텝 {:03d}에서 RMSE: {:.4f}".format(i, accuracy.result().numpy()))

          # print("스텝 {:03d}에서 누적 정확도: {:.4f}".format(i, accuracy.result().numpy()))


    test_accuracy = RootMeanSquaredError()

    y_pred_list = []
    for x, y in test_ds:
        y_pred,w,V = model(x)
#        print(y_pred)
        test_accuracy.update_state(y, y_pred)
        y_pred_list.append(y_pred)

    print("테스트 RMSE: {:.4f}".format(test_accuracy.result().numpy()))
    return y_pred_list, model
```

```python
class FM(tf.keras.Model):
    def __init__(self, p, k):
        super(FM, self).__init__()

        # 모델의 파라미터 정의
        self.w_0 = tf.Variable([0.0])
        self.w = tf.Variable(tf.zeros([p]))
        self.V = tf.Variable(tf.random.normal(shape=(p, k)))

    def call(self, inputs):
        linear_terms = tf.reduce_sum(tf.math.multiply(self.w, inputs), axis=1)

        interactions = 0.5 * tf.reduce_sum(
            tf.math.pow(tf.matmul(inputs, self.V), 2)
            - tf.matmul(tf.math.pow(inputs, 2), tf.math.pow(self.V, 2)),
            1,
            keepdims=False
        )

        y_hat = self.w_0 + linear_terms + interactions
        # return y_hat
        return y_hat, self.w, self.V
```


```python
### 모델링

def func_fm(train_df, test_df):
  # GPU 확인
  tf.config.list_physical_devices('GPU')

  # 자료형 선언
  tf.keras.backend.set_floatx('float32')

  # 데이터 로드
  scaler = MinMaxScaler()
  xtrain = train_df.drop(['rating', 'y','likes'], axis = 1)
  ytrain = train_df.loc[:,'rating']
  xtest = test_df.drop(['rating', 'y', 'likes'], axis = 1)
  ytest = test_df.loc[:,'rating']

  xtrain = scaler.fit_transform(xtrain)
  xtest = scaler.transform(xtest)

  n = xtrain.shape[0] + xtest.shape[0]
  p = xtrain.shape[1]  # 예측 변수의 개수
  k = 12  # 잠재 변수의 개수
  batch_size = 16
  epochs = 10

  # tf.debugging.set_log_device_placement(True)
  # 텐서를 GPU에 할당
  with tf.device('/GPU:0'):
    fm_model = train(xtrain, xtest, ytrain, ytest, p, k, 5, batch=512)

  # fm_model[1].save_weights('/content/drive/MyDrive/temp/fm_check')
  return fm_model, scaler
  
# 모델 저장
# fm_model.load_weights('/content/drive/MyDrive/temp/fm_check')
  
# model[1].save('/content/drive/MyDrive/temp/mymodel')
# tf.keras.models.load_model('/content/drive/MyDrive/temp/mymodel')
```


# 학습한 모델을 이용해 예측하는 함수 생성
```python
def make_pred_list(batch, pro_id, model, test_pred, song_book, scaler, le):
    test_pred_scaled = scaler.transform(test_pred)
    batch = batch
    test_pred_tensor = tf.data.Dataset.from_tensor_slices(
            (tf.cast(test_pred_scaled, tf.float32))).batch(batch)

    rslt_pred = []
    for x in test_pred_tensor:
        rslt_pred.append(model(x)[0])
    rslt = sum([i.numpy().tolist() for i in rslt_pred], [])

    test_pred['pred'] = rslt

    test_pred_dropped = test_pred.drop(['genre','release'], axis=1)
    test_pred_merged = pd.merge(test_pred_dropped, song_book, left_on='song_id', right_on='song_id', how='left')

    p_id = pro_id
    test_pred_merged['Program']=pronumtoname(p_id, le)

    return test_pred_merged
```
