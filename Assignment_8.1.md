
1.Read the following data set


```python
import numpy as np
import pandas as pd
```


```python
Adult=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
```


```python
Adult
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education_num</th>
      <th>marital_status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital_gain</th>
      <th>capital_loss</th>
      <th>hours_per_week</th>
      <th>native_country</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37</td>
      <td>Private</td>
      <td>284582</td>
      <td>Masters</td>
      <td>14</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>5</th>
      <td>49</td>
      <td>Private</td>
      <td>160187</td>
      <td>9th</td>
      <td>5</td>
      <td>Married-spouse-absent</td>
      <td>Other-service</td>
      <td>Not-in-family</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>Jamaica</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>6</th>
      <td>52</td>
      <td>Self-emp-not-inc</td>
      <td>209642</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>7</th>
      <td>31</td>
      <td>Private</td>
      <td>45781</td>
      <td>Masters</td>
      <td>14</td>
      <td>Never-married</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>14084</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>8</th>
      <td>42</td>
      <td>Private</td>
      <td>159449</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>5178</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>9</th>
      <td>37</td>
      <td>Private</td>
      <td>280464</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>10</th>
      <td>30</td>
      <td>State-gov</td>
      <td>141297</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>India</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>11</th>
      <td>23</td>
      <td>Private</td>
      <td>122272</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>12</th>
      <td>32</td>
      <td>Private</td>
      <td>205019</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Never-married</td>
      <td>Sales</td>
      <td>Not-in-family</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>13</th>
      <td>40</td>
      <td>Private</td>
      <td>121772</td>
      <td>Assoc-voc</td>
      <td>11</td>
      <td>Married-civ-spouse</td>
      <td>Craft-repair</td>
      <td>Husband</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>?</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>14</th>
      <td>34</td>
      <td>Private</td>
      <td>245487</td>
      <td>7th-8th</td>
      <td>4</td>
      <td>Married-civ-spouse</td>
      <td>Transport-moving</td>
      <td>Husband</td>
      <td>Amer-Indian-Eskimo</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>Mexico</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>15</th>
      <td>25</td>
      <td>Self-emp-not-inc</td>
      <td>176756</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Farming-fishing</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>35</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>16</th>
      <td>32</td>
      <td>Private</td>
      <td>186824</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Machine-op-inspct</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>17</th>
      <td>38</td>
      <td>Private</td>
      <td>28887</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Sales</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>18</th>
      <td>43</td>
      <td>Self-emp-not-inc</td>
      <td>292175</td>
      <td>Masters</td>
      <td>14</td>
      <td>Divorced</td>
      <td>Exec-managerial</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>19</th>
      <td>40</td>
      <td>Private</td>
      <td>193524</td>
      <td>Doctorate</td>
      <td>16</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>20</th>
      <td>54</td>
      <td>Private</td>
      <td>302146</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Separated</td>
      <td>Other-service</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>21</th>
      <td>35</td>
      <td>Federal-gov</td>
      <td>76845</td>
      <td>9th</td>
      <td>5</td>
      <td>Married-civ-spouse</td>
      <td>Farming-fishing</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>22</th>
      <td>43</td>
      <td>Private</td>
      <td>117037</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Transport-moving</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>2042</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>23</th>
      <td>59</td>
      <td>Private</td>
      <td>109015</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Tech-support</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>24</th>
      <td>56</td>
      <td>Local-gov</td>
      <td>216851</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Tech-support</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>25</th>
      <td>19</td>
      <td>Private</td>
      <td>168294</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Craft-repair</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>26</th>
      <td>54</td>
      <td>?</td>
      <td>180211</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Married-civ-spouse</td>
      <td>?</td>
      <td>Husband</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>South</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>27</th>
      <td>39</td>
      <td>Private</td>
      <td>367260</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>28</th>
      <td>49</td>
      <td>Private</td>
      <td>193366</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Craft-repair</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>29</th>
      <td>23</td>
      <td>Local-gov</td>
      <td>190709</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Never-married</td>
      <td>Protective-serv</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>52</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32530</th>
      <td>30</td>
      <td>?</td>
      <td>33811</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>?</td>
      <td>Not-in-family</td>
      <td>Asian-Pac-Islander</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>99</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32531</th>
      <td>34</td>
      <td>Private</td>
      <td>204461</td>
      <td>Doctorate</td>
      <td>16</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32532</th>
      <td>54</td>
      <td>Private</td>
      <td>337992</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>Japan</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32533</th>
      <td>37</td>
      <td>Private</td>
      <td>179137</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Divorced</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32534</th>
      <td>22</td>
      <td>Private</td>
      <td>325033</td>
      <td>12th</td>
      <td>8</td>
      <td>Never-married</td>
      <td>Protective-serv</td>
      <td>Own-child</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>35</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32535</th>
      <td>34</td>
      <td>Private</td>
      <td>160216</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>55</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32536</th>
      <td>30</td>
      <td>Private</td>
      <td>345898</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Craft-repair</td>
      <td>Not-in-family</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>46</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32537</th>
      <td>38</td>
      <td>Private</td>
      <td>139180</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Divorced</td>
      <td>Prof-specialty</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>15020</td>
      <td>0</td>
      <td>45</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32538</th>
      <td>71</td>
      <td>?</td>
      <td>287372</td>
      <td>Doctorate</td>
      <td>16</td>
      <td>Married-civ-spouse</td>
      <td>?</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32539</th>
      <td>45</td>
      <td>State-gov</td>
      <td>252208</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Separated</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32540</th>
      <td>41</td>
      <td>?</td>
      <td>202822</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Separated</td>
      <td>?</td>
      <td>Not-in-family</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32541</th>
      <td>72</td>
      <td>?</td>
      <td>129912</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>?</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32542</th>
      <td>45</td>
      <td>Local-gov</td>
      <td>119199</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Divorced</td>
      <td>Prof-specialty</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32543</th>
      <td>31</td>
      <td>Private</td>
      <td>199655</td>
      <td>Masters</td>
      <td>14</td>
      <td>Divorced</td>
      <td>Other-service</td>
      <td>Not-in-family</td>
      <td>Other</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32544</th>
      <td>39</td>
      <td>Local-gov</td>
      <td>111499</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Married-civ-spouse</td>
      <td>Adm-clerical</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32545</th>
      <td>37</td>
      <td>Private</td>
      <td>198216</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Divorced</td>
      <td>Tech-support</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32546</th>
      <td>43</td>
      <td>Private</td>
      <td>260761</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Mexico</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32547</th>
      <td>65</td>
      <td>Self-emp-not-inc</td>
      <td>99359</td>
      <td>Prof-school</td>
      <td>15</td>
      <td>Never-married</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>1086</td>
      <td>0</td>
      <td>60</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32548</th>
      <td>43</td>
      <td>State-gov</td>
      <td>255835</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Divorced</td>
      <td>Adm-clerical</td>
      <td>Other-relative</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32549</th>
      <td>43</td>
      <td>Self-emp-not-inc</td>
      <td>27242</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Married-civ-spouse</td>
      <td>Craft-repair</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32550</th>
      <td>32</td>
      <td>Private</td>
      <td>34066</td>
      <td>10th</td>
      <td>6</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Amer-Indian-Eskimo</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32551</th>
      <td>43</td>
      <td>Private</td>
      <td>84661</td>
      <td>Assoc-voc</td>
      <td>11</td>
      <td>Married-civ-spouse</td>
      <td>Sales</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32552</th>
      <td>32</td>
      <td>Private</td>
      <td>116138</td>
      <td>Masters</td>
      <td>14</td>
      <td>Never-married</td>
      <td>Tech-support</td>
      <td>Not-in-family</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>Taiwan</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32553</th>
      <td>53</td>
      <td>Private</td>
      <td>321865</td>
      <td>Masters</td>
      <td>14</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32554</th>
      <td>22</td>
      <td>Private</td>
      <td>310152</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Never-married</td>
      <td>Protective-serv</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32555</th>
      <td>27</td>
      <td>Private</td>
      <td>257302</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Married-civ-spouse</td>
      <td>Tech-support</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32556</th>
      <td>40</td>
      <td>Private</td>
      <td>154374</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32557</th>
      <td>58</td>
      <td>Private</td>
      <td>151910</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32558</th>
      <td>22</td>
      <td>Private</td>
      <td>201490</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32559</th>
      <td>52</td>
      <td>Self-emp-inc</td>
      <td>287927</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>15024</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
<p>32560 rows × 15 columns</p>
</div>



2.Rename the columns as per the description from this file:


```python
Adult.columns=['age','workclass','fnlwgt','education','education_num',
               'marital_status',
               'occupation','relationship','race','sex','capital_gain',
               'capital_loss','hours_per_week','native_country','Amount']

```


```python
Adult.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education_num</th>
      <th>marital_status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital_gain</th>
      <th>capital_loss</th>
      <th>hours_per_week</th>
      <th>native_country</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37</td>
      <td>Private</td>
      <td>284582</td>
      <td>Masters</td>
      <td>14</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>




```python
from pandas import DataFrame, Series
import sqlite3 
from pandasql import sqldf as sql
import sys
```

Create a sql db from adult dataset and name it sqladb


```python
conn = sqlite3.connect('TestDB.db')
```


```python
3.fCreate a sql db from adult dataset and name it sqladb
```


```python
c = conn.cursor()
 
```


```python
Adult.to_sql("Adult.csv", conn, if_exists="replace")

```


```python
conn.execute(
    """
    create table my_table as 
    select * from Adult
    """)

```




    <sqlite3.Cursor at 0x2e1de2740a0>




```python
df = pd.read_sql_query("select * from Adult limit 10;", conn)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education_num</th>
      <th>marital_status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital_gain</th>
      <th>capital_loss</th>
      <th>hours_per_week</th>
      <th>native_country</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>37</td>
      <td>Private</td>
      <td>284582</td>
      <td>Masters</td>
      <td>14</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>49</td>
      <td>Private</td>
      <td>160187</td>
      <td>9th</td>
      <td>5</td>
      <td>Married-spouse-absent</td>
      <td>Other-service</td>
      <td>Not-in-family</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>Jamaica</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>52</td>
      <td>Self-emp-not-inc</td>
      <td>209642</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>31</td>
      <td>Private</td>
      <td>45781</td>
      <td>Masters</td>
      <td>14</td>
      <td>Never-married</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>14084</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>42</td>
      <td>Private</td>
      <td>159449</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>5178</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>37</td>
      <td>Private</td>
      <td>280464</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
</div>



Show me the average hours per week of all men who are working in private sector


```python
df1 = pd.read_sql_query('SELECT hours_per_week,sex, avg(hours_per_week) as `Avg_hours`'
                       'FROM Adult '
                       'GROUP BY sex ', conn)

```


```python
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hours_per_week</th>
      <th>sex</th>
      <th>Avg_hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40</td>
      <td>Female</td>
      <td>36.410361</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>Male</td>
      <td>42.428198</td>
    </tr>
  </tbody>
</table>
</div>



Show me the frequency table for education, occupation and relationship, separately


```python
sql1= """SELECT education,COUNT(*) as cnt
         FROM Adult
         GROUP BY education;"""
df2= pd.read_sql_query(sql1, conn)
```


```python
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>education</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10th</td>
      <td>933</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11th</td>
      <td>1175</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12th</td>
      <td>433</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1st-4th</td>
      <td>168</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5th-6th</td>
      <td>333</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7th-8th</td>
      <td>646</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9th</td>
      <td>514</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Assoc-acdm</td>
      <td>1067</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Assoc-voc</td>
      <td>1382</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bachelors</td>
      <td>5354</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Doctorate</td>
      <td>413</td>
    </tr>
    <tr>
      <th>11</th>
      <td>HS-grad</td>
      <td>10501</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Masters</td>
      <td>1723</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Preschool</td>
      <td>51</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Prof-school</td>
      <td>576</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Some-college</td>
      <td>7291</td>
    </tr>
  </tbody>
</table>
</div>



 occupation:Frequency Table


```python
sql2= """SELECT occupation,COUNT(*) as cnt
         FROM Adult
         GROUP BY occupation;"""
df3= pd.read_sql_query(sql2, conn)
df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>occupation</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>?</td>
      <td>1843</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adm-clerical</td>
      <td>3769</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Armed-Forces</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Craft-repair</td>
      <td>4099</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Exec-managerial</td>
      <td>4066</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Farming-fishing</td>
      <td>994</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Handlers-cleaners</td>
      <td>1370</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Machine-op-inspct</td>
      <td>2002</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Other-service</td>
      <td>3295</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Priv-house-serv</td>
      <td>149</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Prof-specialty</td>
      <td>4140</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Protective-serv</td>
      <td>649</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Sales</td>
      <td>3650</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Tech-support</td>
      <td>928</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Transport-moving</td>
      <td>1597</td>
    </tr>
  </tbody>
</table>
</div>



relationship:Frequency Table


```python
sql3= """SELECT relationship,COUNT(*) as cnt
         FROM Adult
         GROUP BY relationship;"""
df4= pd.read_sql_query(sql3, conn)
df4
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>relationship</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Husband</td>
      <td>13193</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Not-in-family</td>
      <td>8304</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Other-relative</td>
      <td>981</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Own-child</td>
      <td>5068</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Unmarried</td>
      <td>3446</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Wife</td>
      <td>1568</td>
    </tr>
  </tbody>
</table>
</div>



4. Are there any people who are married, working in private sector and having a masters degree


```python
sql4= """SELECT marital_status,education,workclass,COUNT(*) as cnt
         FROM Adult
         Where education="Masters" & workclass="Private";"""
df5= pd.read_sql_query(sql4, conn)
df5
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>marital_status</th>
      <th>education</th>
      <th>workclass</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



5. What is the average, minimum and maximum age group for people working in different sectors


```python
sql5= """SELECT occupation,min(age) as min_age,max(age) as max_age, avg(age) as avg_age
         FROM Adult
         GROUP BY occupation;"""
df6= pd.read_sql_query(sql5, conn)
df6
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>occupation</th>
      <th>min_age</th>
      <th>max_age</th>
      <th>avg_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>?</td>
      <td>17</td>
      <td>90</td>
      <td>40.882800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adm-clerical</td>
      <td>17</td>
      <td>90</td>
      <td>36.963916</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Armed-Forces</td>
      <td>23</td>
      <td>46</td>
      <td>30.222222</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Craft-repair</td>
      <td>17</td>
      <td>90</td>
      <td>39.031471</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Exec-managerial</td>
      <td>17</td>
      <td>90</td>
      <td>42.169208</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Farming-fishing</td>
      <td>17</td>
      <td>90</td>
      <td>41.211268</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Handlers-cleaners</td>
      <td>17</td>
      <td>90</td>
      <td>32.165693</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Machine-op-inspct</td>
      <td>17</td>
      <td>90</td>
      <td>37.715285</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Other-service</td>
      <td>17</td>
      <td>90</td>
      <td>34.949621</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Priv-house-serv</td>
      <td>17</td>
      <td>81</td>
      <td>41.724832</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Prof-specialty</td>
      <td>17</td>
      <td>90</td>
      <td>40.517633</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Protective-serv</td>
      <td>17</td>
      <td>90</td>
      <td>38.953775</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Sales</td>
      <td>17</td>
      <td>90</td>
      <td>37.353973</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Tech-support</td>
      <td>17</td>
      <td>73</td>
      <td>37.022629</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Transport-moving</td>
      <td>17</td>
      <td>90</td>
      <td>40.197871</td>
    </tr>
  </tbody>
</table>
</div>



6. Calculate age distribution by country


```python
sql6= """SELECT age,native_country,count(*) as cnt
         FROM Adult
         GROUP BY native_country;"""
df7= pd.read_sql_query(sql6, conn)
df7
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>native_country</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>81</td>
      <td>?</td>
      <td>583</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48</td>
      <td>Cambodia</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32</td>
      <td>Canada</td>
      <td>121</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>China</td>
      <td>75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49</td>
      <td>Columbia</td>
      <td>59</td>
    </tr>
    <tr>
      <th>5</th>
      <td>48</td>
      <td>Cuba</td>
      <td>95</td>
    </tr>
    <tr>
      <th>6</th>
      <td>45</td>
      <td>Dominican-Republic</td>
      <td>70</td>
    </tr>
    <tr>
      <th>7</th>
      <td>27</td>
      <td>Ecuador</td>
      <td>28</td>
    </tr>
    <tr>
      <th>8</th>
      <td>39</td>
      <td>El-Salvador</td>
      <td>106</td>
    </tr>
    <tr>
      <th>9</th>
      <td>48</td>
      <td>England</td>
      <td>90</td>
    </tr>
    <tr>
      <th>10</th>
      <td>64</td>
      <td>France</td>
      <td>29</td>
    </tr>
    <tr>
      <th>11</th>
      <td>74</td>
      <td>Germany</td>
      <td>137</td>
    </tr>
    <tr>
      <th>12</th>
      <td>23</td>
      <td>Greece</td>
      <td>29</td>
    </tr>
    <tr>
      <th>13</th>
      <td>22</td>
      <td>Guatemala</td>
      <td>64</td>
    </tr>
    <tr>
      <th>14</th>
      <td>29</td>
      <td>Haiti</td>
      <td>44</td>
    </tr>
    <tr>
      <th>15</th>
      <td>32</td>
      <td>Holand-Netherlands</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>30</td>
      <td>Honduras</td>
      <td>13</td>
    </tr>
    <tr>
      <th>17</th>
      <td>28</td>
      <td>Hong</td>
      <td>20</td>
    </tr>
    <tr>
      <th>18</th>
      <td>47</td>
      <td>Hungary</td>
      <td>13</td>
    </tr>
    <tr>
      <th>19</th>
      <td>23</td>
      <td>India</td>
      <td>100</td>
    </tr>
    <tr>
      <th>20</th>
      <td>42</td>
      <td>Iran</td>
      <td>43</td>
    </tr>
    <tr>
      <th>21</th>
      <td>23</td>
      <td>Ireland</td>
      <td>24</td>
    </tr>
    <tr>
      <th>22</th>
      <td>41</td>
      <td>Italy</td>
      <td>73</td>
    </tr>
    <tr>
      <th>23</th>
      <td>22</td>
      <td>Jamaica</td>
      <td>81</td>
    </tr>
    <tr>
      <th>24</th>
      <td>54</td>
      <td>Japan</td>
      <td>62</td>
    </tr>
    <tr>
      <th>25</th>
      <td>29</td>
      <td>Laos</td>
      <td>18</td>
    </tr>
    <tr>
      <th>26</th>
      <td>43</td>
      <td>Mexico</td>
      <td>643</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>Nicaragua</td>
      <td>34</td>
    </tr>
    <tr>
      <th>28</th>
      <td>47</td>
      <td>Outlying-US(Guam-USVI-etc)</td>
      <td>14</td>
    </tr>
    <tr>
      <th>29</th>
      <td>25</td>
      <td>Peru</td>
      <td>31</td>
    </tr>
    <tr>
      <th>30</th>
      <td>39</td>
      <td>Philippines</td>
      <td>198</td>
    </tr>
    <tr>
      <th>31</th>
      <td>85</td>
      <td>Poland</td>
      <td>60</td>
    </tr>
    <tr>
      <th>32</th>
      <td>48</td>
      <td>Portugal</td>
      <td>37</td>
    </tr>
    <tr>
      <th>33</th>
      <td>33</td>
      <td>Puerto-Rico</td>
      <td>114</td>
    </tr>
    <tr>
      <th>34</th>
      <td>49</td>
      <td>Scotland</td>
      <td>12</td>
    </tr>
    <tr>
      <th>35</th>
      <td>22</td>
      <td>South</td>
      <td>80</td>
    </tr>
    <tr>
      <th>36</th>
      <td>32</td>
      <td>Taiwan</td>
      <td>51</td>
    </tr>
    <tr>
      <th>37</th>
      <td>55</td>
      <td>Thailand</td>
      <td>18</td>
    </tr>
    <tr>
      <th>38</th>
      <td>32</td>
      <td>Trinadad&amp;Tobago</td>
      <td>19</td>
    </tr>
    <tr>
      <th>39</th>
      <td>52</td>
      <td>United-States</td>
      <td>29169</td>
    </tr>
    <tr>
      <th>40</th>
      <td>51</td>
      <td>Vietnam</td>
      <td>67</td>
    </tr>
    <tr>
      <th>41</th>
      <td>29</td>
      <td>Yugoslavia</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
