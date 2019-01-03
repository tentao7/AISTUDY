
# coding: utf-8

# ## 04. 대표적 비지도 학습법 
#  * Autoencoder

# ## 학습 내용
# ### 01. Autoencoder란?
# ### 02. 간단한 예제를 보자.
# ### 03. 왜 사용되는가?

# ### 01. Autoencoder란?

# In[1]:


### 대표적인 비지도학습으로 많이 쓰이는 신경망 오토인코더(Autoencoder)가 있다.


# ### 오토 인코더는 입력값과 출력값을 같게 하는 신경망이다.

# ### 가운데 계층의 노드 수가 입력값보다 적은 것이 특징이다.

# ### 결과적으로 입력 데이터를 압축하는 효과를 얻는다. 이 과정이 노이즈 제거에 많이 효과적이다.

# ## 핵심 메모 : 
#  *  (01) 입력층으로 들어온 데이터를 인코더를 통해 은닉층으로 내보낸다.
#  *  (02) 은닉층의 데이터를 디코더를 통해 출력층으로 내보낸다.
#  *  (03) 만들어진 출력값과 입력값이 같아지도록 만드는 가중치를 찾아낸다.

# ### 02. 간단한 예제를 보자.

# In[2]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


# In[5]:


learning_rate = 0.01    # 학습율 0.01
training_epoch = 20     # 훈련 횟수 20회 
batch_size = 100        # 배치 사이즈 100
n_hidden = 256          # 은닉층의 개수 256
n_input = 28 * 28       # 784개 (입력층) 


# ### 인코더 만들기 
#  * 맨처음은 n_hidden개의 뉴런을 만든다.
#  * 가중치와 편향 변수를 원하는 뉴런의 개수만큼 설정한다.
#  * 활성화 함수 sigmoid 함수를 적용한다.

# In[7]:


X = tf.placeholder(tf.float32, [None, n_input])
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))

encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))


# ### 디코더 만들기

# In[8]:


W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_decode))


# In[9]:


cost = tf.reduce_mean(tf.pow(X- decoder, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


# ### 학습을 진행

# In[10]:


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(training_epoch):
    total_cost = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], 
                              feed_dict={X:batch_xs})
        total_cost += cost_val
        
    print('Epoch:', '%04d' % (epoch + 1), 
         'Avg. cost=', '{:.4f}'.format(total_cost / total_batch))
    
print('최적화 완료!')


# ### 총 10개의 테스트 데이터를 가져와 디코더를 이용해 출력값을 만든다.

# In[11]:


sample_size = 10
samples = sess.run(decoder, 
                  feed_dict = {X:mnist.test.images[:sample_size]})


# ## 위쪽이 원본 데이터, 아래쪽이 신경망이 생성한 이미지

# In[14]:


fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
    ax[1][i].imshow(np.reshape(samples[i], (28,28)))
    
plt.show()

