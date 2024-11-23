#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# İhtiyacım olacak kütüphaneleri içe aktarma işlemini gerçekleştirdim.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df_ = pd.read_csv('/kaggle/input/-spotify-tracks-dataset/dataset.csv')
df = df_.copy()


# In[4]:


# Veri setinde rasgele eksik veriler oluşturalım.

import random

def add_random_missing_values(dataframe: pd.DataFrame,
                              missing_rate: float = 0.05,
                              seed: random = 42) -> pd.DataFrame:
    """Turns random values to NaN in a DataFrame.
    
    To use this function, you need to import pandas, numpy and random libraries.

    Args:
        dataframe (pd.DataFrame): DataFrame to be processed.
        missing_rate (float): Percentage of missing value rate in float format. Defaults 0.05

    
    """
    # Get copy of dataframe
    df_missing = dataframe.copy()

    # Obtain size of dataframe and number total number of missing values
    df_size = dataframe.size
    num_missing = int(df_size * missing_rate)
    
    # Set seed
    if seed:
        random.seed(seed)

    # Get random row and column indexes to turn them NaN
    for _ in range(num_missing):
        row_idx = random.randint(0, dataframe.shape[0] - 1)
        col_idx = random.randint(0, dataframe.shape[1] - 1)

        df_missing.iat[row_idx, col_idx] = np.nan
        
    return df_missing

df = add_random_missing_values(dataframe = df,
                               missing_rate = 0.03)


# In[5]:


df.head() # Verinin içeriğini bir dataframe halide görmek ve incelemek iç.in head komutunu kullandım.


# In[6]:


df.info() # Veri setinin sütunlarını, satır sayılarını, veri tiplerini ve kayıp olmayan veri sayısını görmek için info komutunu kullandım.


# In[7]:


df.shape # shape atribütü ile veri setinin boyutunu kontrol ettim.


# In[8]:


print(list(df.columns), '\n') # Veri setimde olan sütun isimlerini yani özellikleri liste halinde inceledim.


# In[30]:


categorical_features = [col for col in df.columns if (df[col].dtype == 'object') or (df[col].dtype == 'category')]
numerical_features = [col for col in df.columns if (df[col].dtype == 'float64') or (df[col].dtype == 'int')]

print('Kategorik Özellikler:\n\n','\n'.join(categorical_features),sep='')
print('\n-----------------------\n')
print('Sayısal Özellikler:\n\n','\n'.join(numrical_features),sep='')

# Burada kategorik ve sayısal özelliklerimi kendi arasında listeledim ve daha okunabilir şekilde görmek için düzenlemeler yaptım.


# In[10]:


df = df.drop('Unnamed: 0', axis=1) # Unnamed özelliği aslında index ile aynı görevi gördüğünden ve veri analizinde kullanılmayacağından veri setimden çıkardım.


# In[12]:


df.isnull().sum() # Özellikler özelinde kayıp veri sayımı görmek için boş değer olup olmama durumuna bakarak olmayanların sayısını veren kodu kullandım.


# - Kategorikveri türüyle çalışmanın avantajlarını gerektiğinde kullanabilmek için baştan tüm string veri tiplerindeki özellikleri kategori veri tipine çevireceğim.

# In[13]:


for col in df.columns:
    if col in categorical_features:
        df[col] = pd.Categorical(df[col])


# In[14]:


df.info()


# In[55]:


# Veri seti içerisindeki sayısal özelliklere ait verilerin sahip olduğu istatistiksel bilgilere göz atmak ve veri hakkında daha fazla bilgiye sahip olmak için kullanıyorum.

df.describe().T


# - İlk olarak veriler arasından loudness özelliğinde çoğu yerde sonuçlar negatif değerlere sahipken ortalama ve verilerin dağılımına bakarsak standart sapmanın yüksek olduğunu, min ve max noktalarının çok geniş bir aralığı işaret ettiğine bakarak bu özellikte aykırı değerlerin olabileceğini görebiliyorum.

# In[16]:


print(f"Veri seti içerisinde toplam {df.notnull().sum().sum()} adet eksik olmayan, {df.isnull().sum().sum()} eksik gözlem var.")


# In[17]:


df[df.isnull().any(axis = 1)] # en az bir hücresi eksik olan satırlar ve sayısı


# In[18]:


df[df.notnull().all(axis = 1)] # hiç eksik verisi olmayan satırlar ve sayısı


# In[19]:


# Aşağıda şarkıların popularitesini etkileyen özelliklerin sıralamasını ve korelasyonunu veren bir grafik ve bu grafiği oluşturduğum kod var.

# Sayısal sütunları seçme
df_numeric = df.select_dtypes(include=['number'])

corr = df_numeric.corrwith(df_numeric['popularity']).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
sns.heatmap(pd.DataFrame(corr , columns=['correlation']), annot=True, cmap='coolwarm', fmt=".4f", linewidths=.5)
plt.title('Popüleriteye Göre Korelasyon')
plt.show()


# In[24]:


# Sanatçı, albüm ve şarkı isimlerinde benzerlik ve çalışmalardan doğabilecek yanlış veri oluşmasını göz önüne alarak bu sütunlardan herhangi biri boş ise o satırları veri setinden
# kaldıracağım

df = df.dropna(subset=['artists', 'album_name', 'track_name'])


# In[25]:


df.isnull().sum()


# In[39]:


# Burada tüm sayısal özelliklerin histplot dağılım grafiklerini aynanda çağırarak aynanda tüm sayısal özelliklerin dağılımına göz atarak doldurmada nasıl bir yol izleyeceğime kara vereceğim.

df_numeric = df.select_dtypes(include=['float64'])

df_numeric.hist(bins=30, figsize=(10, 10))
plt.show()

# Çarpıklık değerini hesaplama
for col in df_numeric.columns:
    skewness = df[col].skew()  # Çarpıklık hesaplama
    print(f"{col} - Çarpıklık: {skewness}")

# Ayrıca çarpıklık değerlerini de alta yazarak çarpıklık değerine göre de inceleme gerçekleştiriyorum.


# Bu aşamada ayrıca aykırı değerlerin dağılımı ve miktarı hakkında fikir sahibi olmak için boxplot grafiğini kullanarak tüm sayısal verilerin dağılımlarını inceleyec.eğim

# In[38]:


# Grafik boyutlarını ayarlama
num_columns = df_numeric.shape[1]
fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(10, 5*num_columns))

# Eğer sadece bir sütun varsa axes'i bir listeye çevirmeliyiz. Bunun nedeni her sütunu kendi içinde tek bir grafik olarak gösterebilmek için
# Aslında aşağıdaki sıralı grafik gösterimini bu sayede gerçekleştirdim.
if num_columns == 1:
    axes = [axes]

# Her bir sütun için boxplot çizme
for i, column in enumerate(df_numeric.columns):
    sns.boxplot(data=df_numeric[column], ax=axes[i], palette="Set2")
    axes[i].set_title(f'{column} - Boxplot', fontsize=14)
plt.tight_layout()
plt.show()


# In[40]:


# Aykırı değerleri olmayan ya da çok az olan sayısal verileri ortalama ile dolduracağım.
columns_to_fill_mean = ['popularity', 'energy', 'key', 'mode', 'acousticness', 'valence']

# Her bir sütunun ortalama değeri ile eksik verileri dolduracağım.
for column in columns_to_fill_mean:
    df[column] = df[column].fillna(df[column].mean())


# In[41]:


# İşlemden sonra çalıştığım sütunlarda hala eksik veri var mı, kod düzgün çalışıyor mu diye kontrol edelim.

df.isnull().sum()


# In[42]:


# Bu sefer aykırı değer sahip olan sütunları medyanı ile dolduruyorum ki aykırı değerlerden etkilenme olmadan daha verimli bir doldurma işlemi gerçekleşsin
columns_to_fill_median = ['duration_ms', 'danceability', 'loudness', 'speechiness', 'instrumentalness', 'liveness', 'tempo', 'time_signature']

# Her bir sütunun ortalama değeri ile eksik verileri doldurmak
for column in columns_to_fill_median:
    df[column] = df[column].fillna(df[column].median())


# In[43]:


df.isnull().sum()


# In[45]:


df_categorical = df.select_dtypes(include = "category")
df_categorical.head(3)


# In[48]:


# Kategorik verilerden explicit ve track_genre sütunlarını modları ile doldurmayı tercih ettim. Pratik ve etkili bir yöntem olduğu için bunu seçtim.
columns_to_fill_cat_mode = ['explicit', 'track_genre']

# Her bir sütunun ortalama değeri ile eksik verileri doldurmak
for column in columns_to_fill_cat_mode:
    df[column] = df[column].fillna(df[column].mode()[0])


# In[49]:


df.isnull().sum()


# In[51]:


print(len('5SuOikwiRyPMVoIQDJUgSV'))
print(len('4qPNDBW1i3p13qLCt0Ki3A'))
print(len('1iJBSr7s7jYXzM8EGcbK5b'))


# - Burada tüm idlerin uzunluğunun aynı olup olmadığına baktım.Bunu gözeterek boş değerleri rastgele idler üreten bir fonksiyon ile dolduracağım.

# In[53]:


# track_id sütunundaki veriler aslında unique değerler. Bu veri seti Spotify bünyesindeki parçaları ve özelliklerini içerdiğinden her parça benzersiz.
# Bu nedenle id'ler de benzersiz olmak zorunda olduğundan bu sütun için diğer doldurma seçeneklerinin dışında yeni benzersiz id'ler üretmek zorundaydım.

import random
import string

# track_id'yi string veri tipine dönüştürdüm.Çünkü category veri tipinde hata veriyordu ya veriyi yeni bir dataframede category olark toplayacaktım ya da 
# tekrar object veri tipine alıp öyle değiştirecektim.Bu yöntem daha kolay ve hızlı olduğundan bu yöntemi seçtim.
df['track_id'] = df['track_id'].astype(str)

# Rastgele ve benzersiz track_id üretmek için fonksiyon
def generate_custom_track_id(length=22):
    # Belirtilen uzunluktaki rastgele alfanümerik ID üretme
    characters = string.ascii_letters + string.digits  # Harfler ve rakamlar: burada hem harf hem rakamlardan bir karakter oluşturmak için bu yöntemi buldum ve kullandım.
    return ''.join(random.choice(characters) for i in range(length))

# yukarıda oluşturduğum fonksiyonu kullanarak boş hücrelere yeni üretilmiş idler ile doldurdum.
df['track_id'] = df['track_id'].fillna(df.apply(lambda x: generate_custom_track_id(), axis=1))


# In[54]:


df.isnull().sum()


# Verilerin karakteristiği ve dağılımlarına bakarak yaptığım analizler ile verinin tüm eksik verilerini doldurarak veri setini kullanıma hazır hale getirdim.

# - Bu veri setinde aslında birçok farklı özelliğin o parçanın popülaritesini etkilediğini ve bunların hangi düzeyde olduğunu gördük.
# - Buradan yola çıkarak popülaritesi yüksek olan parçaların türüne bakarak aslında en çok hangi türlerin tercih edildiğini gözlemleyebiliriz.
# - Bunu kullanarak uygun bir yapay zeka modellemesiyle bir tür içerisindeki en popüler parçaları o türü en çok dinleye dinleyicilere öneren bir model geliştirebiliriz.
# - Ayrıca tüm dinleyicilere o andaki en popüler şarkılardan oluşan bir içerik ya da türlere göre farklı içerikler ve öneriler üreten bir model de oluşturabiliriz.
# - Anlık olarak buraya gelecek yeni verilerideki popülariteyi takip ederek dinleyici kitlesinin hangi yöne doğru gittiğini takipte edebiliriz.
# - Bu inceleme aynı zamanda pazara da fikir olarak dönebilir.

# In[ ]:




