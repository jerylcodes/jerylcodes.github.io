---
layout: single
permalink: proj3/
title: &title "Project 3"
author_profile: true
---

<img src="https://i.imgur.com/iYGFYFv.png" style="float: left; margin: 15px; height: 80px">

### Web Scraping and Analysis
---
For this project, we will explore web scraping to obtain data from websites. More specifically, we will be looking at job search websites to obtain job listings together with job descriptions and salary information. We will next use the information scraped to predict salary or classify jobs according to its description. Classifying jobs according to job description involves analysing textual information. We can make use of scikit learn's natural language processing packages to help us in our analysis. 

Packages used:
1. BeautifulSoup
2. pandas
3. urllib3
4. re
5. pickle
6. matplotlib
7. seaborn
8. scikit-learn
9. imblearn

<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

### 1.1 Getting the links from the job site for scraping using BeautifulSoup and urllib3 


```python
url = 'https://jobscentral.com.sg/jobsearch?q=data%20science&pg='
urllist = []
for i in range(1,15):
    a = url + str(i)
    urllist.append(a)
```


```python
links = []

for link in urllist:
    http = urllib3.PoolManager()
    response = http.request('GET', link)
    soup = BeautifulSoup(response.data, "html.parser")
    
    a = soup.find_all('a',attrs={"class":"job-title"}, href=True)

    for val in a:
        linkstr = 'https://jobscentral.com.sg' + val['href']
        if linkstr not in links:
            links.append(linkstr)
```

    C:\ProgramData\Anaconda2\envs\py36\lib\site-packages\urllib3\connectionpool.py:858: InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
      InsecureRequestWarning)
    C:\ProgramData\Anaconda2\envs\py36\lib\site-packages\urllib3\connectionpool.py:858: InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
      InsecureRequestWarning)
    C:\ProgramData\Anaconda2\envs\py36\lib\site-packages\urllib3\connectionpool.py:858: InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
      InsecureRequestWarning)
    C:\ProgramData\Anaconda2\envs\py36\lib\site-packages\urllib3\connectionpool.py:858: InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
      InsecureRequestWarning)
    C:\ProgramData\Anaconda2\envs\py36\lib\site-packages\urllib3\connectionpool.py:858: InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
      InsecureRequestWarning)
    C:\ProgramData\Anaconda2\envs\py36\lib\site-packages\urllib3\connectionpool.py:858: InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
      InsecureRequestWarning)
    C:\ProgramData\Anaconda2\envs\py36\lib\site-packages\urllib3\connectionpool.py:858: InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
      InsecureRequestWarning)
    C:\ProgramData\Anaconda2\envs\py36\lib\site-packages\urllib3\connectionpool.py:858: InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
      InsecureRequestWarning)
    C:\ProgramData\Anaconda2\envs\py36\lib\site-packages\urllib3\connectionpool.py:858: InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
      InsecureRequestWarning)
    C:\ProgramData\Anaconda2\envs\py36\lib\site-packages\urllib3\connectionpool.py:858: InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
      InsecureRequestWarning)
    C:\ProgramData\Anaconda2\envs\py36\lib\site-packages\urllib3\connectionpool.py:858: InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
      InsecureRequestWarning)
    C:\ProgramData\Anaconda2\envs\py36\lib\site-packages\urllib3\connectionpool.py:858: InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
      InsecureRequestWarning)
    C:\ProgramData\Anaconda2\envs\py36\lib\site-packages\urllib3\connectionpool.py:858: InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
      InsecureRequestWarning)
    C:\ProgramData\Anaconda2\envs\py36\lib\site-packages\urllib3\connectionpool.py:858: InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
      InsecureRequestWarning)
    

### 1.2 Append results into pandas dataframe


```python
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
      <th>Title</th>
      <th>Company</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Senior/Data Science Engineer</td>
      <td>Micron Semiconductor Asia Date Posted: 28-Jan-...</td>
      <td>[&lt;p&gt;&lt;/p&gt;, &lt;p&gt;Req. ID: 94680 &lt;/p&gt;, &lt;p&gt;&lt;strong&gt;J...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BIG DATA ENGINEER</td>
      <td>Micron Semiconductor Asia Date Posted: 15-Jan-...</td>
      <td>[&lt;p&gt;&lt;/p&gt;, &lt;p&gt;Req. ID: 88661 &lt;/p&gt;, &lt;p&gt;&lt;strong&gt;R...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TPCE Data Analytics Engineer</td>
      <td>Micron Semiconductor Asia Date Posted: 26-Dec-...</td>
      <td>[&lt;p&gt;&lt;/p&gt;, &lt;p&gt;Req. ID: 104841 &lt;/p&gt;, &lt;p&gt;The cand...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Project Manager – Analytics &amp; Data Management</td>
      <td>Optimum Solutions (S) Pte Ltd Date Posted: 5-J...</td>
      <td>[&lt;p&gt;&lt;/p&gt;, &lt;p&gt; &lt;span&gt;Responsibilities&lt;br/&gt;
&lt;br/...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VP/AVP, Data Engineer - Corporate Treasury, Gr...</td>
      <td>DBS Bank Ltd Date Posted: 12-Dec-2017</td>
      <td>[&lt;p&gt;&lt;b&gt;Business Function&lt;/b&gt;&lt;/p&gt;, &lt;p&gt;&lt;/p&gt;, &lt;p&gt;...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>VP/AVP, Audit Digitalization Lead - Data Analy...</td>
      <td>DBS Bank Ltd Date Posted: 14-Jan-2018</td>
      <td>[&lt;p&gt;&lt;b&gt;Business Function&lt;/b&gt; &lt;/p&gt;, &lt;p&gt;&lt;/p&gt;, &lt;p...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SERVICE DELIVERY SPECIALIST (SHIFT WORK) (DATA...</td>
      <td>The Search Executives Pte Ltd (EA Licence No: ...</td>
      <td>[&lt;p&gt;&lt;span&gt;&lt;span&gt;&lt;br/&gt;
Must Have:&lt;/span&gt;&lt;br/&gt;
&lt;...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CRITICAL INFRA SOLUTIONING &amp; PROVISIONING (CIS...</td>
      <td>The Search Executives Pte Ltd (EA Licence No: ...</td>
      <td>[&lt;p&gt;&lt;span&gt;&lt;br/&gt;
&lt;span&gt;Good To Have:&lt;br/&gt;
&lt;br/&gt;...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SERVICE DELIVERY &amp; NOC TEAM LEAD (DATA CENTRE)...</td>
      <td>The Search Executives Pte Ltd (EA Licence No: ...</td>
      <td>[&lt;p&gt;&lt;span&gt; &lt;/span&gt;&lt;/p&gt;, &lt;p&gt;&lt;span&gt;Requirements:...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>System Analyst (Data Profiling)</td>
      <td>MavenTree Technology Pte Ltd Date Posted: 20-J...</td>
      <td>[&lt;p&gt;&lt;br/&gt;&lt;/p&gt;, &lt;p&gt;&lt;br/&gt;&lt;/p&gt;, &lt;p _ngcontent-c2=...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Looking for a Big Data Developer for a Permane...</td>
      <td>Aryan Search Pte Ltd Date Posted: 9-Jan-2018</td>
      <td>[&lt;p&gt;&lt;/p&gt;, &lt;p&gt; &lt;span&gt;We are looking for a &lt;stro...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Data Center Operator</td>
      <td>ServLink Technology Resources Pte Ltd Date Pos...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Senior/ Data Scientist (KDM/ILS/OBH1)</td>
      <td>Singapore Technologies Kinetics Ltd Date Poste...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Senior/ Data Scientist (KDM/ILS/OBH1)</td>
      <td>Singapore Technologies Kinetics Ltd Date Poste...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Senior/ Data Scientist (KDM/ILS/OBH1)</td>
      <td>Singapore Technologies Kinetics Ltd Date Poste...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Data Analyst (EAST/IMMEDIATE)</td>
      <td>Adecco Personnel Pte Ltd Date Posted: 27-Jan-2018</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Data Analyst</td>
      <td>Illumina Singapore Pte Ltd Date Posted: 12-Jan...</td>
      <td>[&lt;p&gt;&lt;b&gt;Position Summary:&lt;/b&gt;&lt;br/&gt;&lt;br/&gt;This pos...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Senior / Software Developers  (AI, Big Data, E...</td>
      <td>THATZ International Pte Ltd Date Posted: 19-Ja...</td>
      <td>[&lt;p&gt;&lt;strong&gt;&lt;span&gt;Key Area of Responsibilities...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Data Integration Development Support</td>
      <td>Aryan Search Pte Ltd Date Posted: 15-Jan-2018</td>
      <td>[&lt;p&gt;&lt;/p&gt;, &lt;p&gt; &lt;span&gt;We are looking for a &lt;stro...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Assistant / Deputy Manager, Data Analysis</td>
      <td>Land Transport Authority (LTA) Date Posted: 29...</td>
      <td>[&lt;p&gt;&lt;span&gt;You will derive policy insights from...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Facilities Manager (Data Center)</td>
      <td>Newtech Technology (South Asia) Pte Ltd Date P...</td>
      <td>[&lt;p&gt;&lt;strong&gt;Responsibilities&lt;/strong&gt; &lt;/p&gt;, &lt;p...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Job Opportunity for Network Data Architect</td>
      <td>Aryan Search Pte Ltd Date Posted: 23-Jan-2018</td>
      <td>[&lt;p&gt;&lt;strong&gt;Role: Network Data Architect&lt;/stro...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Data Scientist</td>
      <td>Micron Semiconductor Asia Date Posted: 17-Jan-...</td>
      <td>[&lt;p&gt;Req. ID: 84487 &lt;/p&gt;, &lt;p&gt;&lt;strong&gt;Responsibi...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Fab10 Data Scientist</td>
      <td>Micron Semiconductor Asia Date Posted: 11-Dec-...</td>
      <td>[&lt;p&gt;Req. ID: 103043 &lt;/p&gt;, &lt;p&gt;&lt;strong&gt;Responsib...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>MCT Big Data Senior Engineer</td>
      <td>Micron Semiconductor Asia Date Posted: 18-Jan-...</td>
      <td>[&lt;p&gt;&lt;/p&gt;, &lt;p&gt;Req. ID: 86425 &lt;/p&gt;, &lt;p&gt;&lt;strong&gt;M...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>MCT Big Data Senior Engineer</td>
      <td>Micron Semiconductor Asia Date Posted: 16-Jan-...</td>
      <td>[&lt;p&gt;&lt;strong&gt;&lt;/strong&gt;&lt;/p&gt;, &lt;p&gt;Req. ID: 86425 &lt;...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>HR DATA ANALYST</td>
      <td>Micron Semiconductor Asia Date Posted: 27-Jan-...</td>
      <td>[&lt;p&gt;&lt;/p&gt;, &lt;p&gt;Req. ID: 94577 &lt;/p&gt;, &lt;p&gt;Do you be...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>CAMPUS SECURITY &amp; ADMIN SUPPORT EXECUTIVE (DAT...</td>
      <td>The Search Executives Pte Ltd (EA Licence No: ...</td>
      <td>[&lt;p&gt;&lt;span&gt;We are looking for suitable candidat...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>RDA/Metrology and Big Data Manager</td>
      <td>Micron Semiconductor Asia Date Posted: 3-Jan-2018</td>
      <td>[&lt;p&gt;&lt;/p&gt;, &lt;p&gt;Req. ID: 105627 &lt;/p&gt;, &lt;p&gt;&lt;strong&gt;...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Transport Analyst - Science Industry ($3.5K to...</td>
      <td>Recruit Express Pte Ltd Date Posted: 24-Jan-2018</td>
      <td>[&lt;p&gt;&lt;span&gt;&lt;strong&gt;Key Job Activities&lt;/strong&gt;&lt;...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>298</th>
      <td>Product Engineer (Healthcare) (KXR/RSE/VL3)</td>
      <td>Singapore Technologies Kinetics Ltd Date Poste...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>299</th>
      <td>EXECUTIVE ENGINEER (Digitalisation and IOT sys...</td>
      <td>SIA Engineering Company Date Posted: 19-Dec-2017</td>
      <td>[&lt;p&gt;&lt;/p&gt;, &lt;p&gt;&lt;strong&gt; &lt;/strong&gt;&lt;/p&gt;, &lt;p&gt;&lt;span&gt;...</td>
    </tr>
    <tr>
      <th>300</th>
      <td>SEO/SEM (Search Engine Marketing Specialist)</td>
      <td>THE ADVERTISER Date Posted: 13-Dec-2017</td>
      <td>[&lt;p&gt;&lt;strong&gt;&lt;span&gt;Job Overview:&lt;/span&gt;&lt;/strong...</td>
    </tr>
    <tr>
      <th>301</th>
      <td>Supplier Quality Engineer (KDM/SCM/WC3)</td>
      <td>Singapore Technologies Kinetics Ltd Date Poste...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>302</th>
      <td>System Engineer (Autonomous Vehicle) (KXR/RSE/SY)</td>
      <td>Singapore Technologies Kinetics Ltd Date Poste...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>303</th>
      <td>Troubleshooting Engineer (KDM/ILS/KW1)</td>
      <td>Singapore Technologies Kinetics Ltd Date Poste...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>304</th>
      <td>EXECUTIVE ENGINEER (Information Systems)</td>
      <td>SIA Engineering Company Date Posted: 19-Dec-2017</td>
      <td>[&lt;p&gt;&lt;br/&gt;&lt;/p&gt;, &lt;p&gt;&lt;strong&gt;Requirements&lt;br/&gt;
 &lt;...</td>
    </tr>
    <tr>
      <th>305</th>
      <td>Optical Systems Engineer</td>
      <td>Finisar Singapore Pte Ltd Date Posted: 18-Dec-...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>306</th>
      <td>Systems Engineer</td>
      <td>Cycle &amp; Carriage Singapore Date Posted: 4-Jan-...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>307</th>
      <td>Troubleshooting Engineer (KDM/ILS/KW1)</td>
      <td>Singapore Technologies Kinetics Ltd Date Poste...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>308</th>
      <td>System Engineer (Autonomous Vehicle) (KXR/RSE/SY)</td>
      <td>Singapore Technologies Kinetics Ltd Date Poste...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>309</th>
      <td>Robotics Engineer  (Shared Services) (KXR/STAI...</td>
      <td>STA Inspection Pte Ltd (STA) Date Posted: 7-De...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>310</th>
      <td>STC Senior Engineer - Photolithography</td>
      <td>Micron Semiconductor Asia Date Posted: 3-Jan-2018</td>
      <td>[&lt;p&gt;&lt;/p&gt;, &lt;p&gt;Req. ID: 95091 &lt;/p&gt;, &lt;p&gt;&lt;strong&gt;D...</td>
    </tr>
    <tr>
      <th>311</th>
      <td>Product Engineer (Healthcare) (KXR/RSE/VL3)</td>
      <td>Singapore Technologies Kinetics Ltd Date Poste...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>312</th>
      <td>Order &amp; Inventory Management Executive (KIS/AS...</td>
      <td>Singapore Technologies Kinetics Ltd Date Poste...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>313</th>
      <td>Senior Manager (CBG/SGBC/LWL)</td>
      <td>Singapore Technologies Kinetics Ltd Date Poste...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>314</th>
      <td>APD Singapore Equipment Development Sr Engineer</td>
      <td>Micron Semiconductor Asia Date Posted: 31-Dec-...</td>
      <td>[&lt;p&gt;&lt;/p&gt;, &lt;p&gt;Req. ID: 105201 &lt;/p&gt;, &lt;p&gt;&lt;strong&gt;...</td>
    </tr>
    <tr>
      <th>315</th>
      <td>Assistant Principal Engineer/ Senior Engineer ...</td>
      <td>Innosparks Pte Ltd Date Posted: 3-Dec-2017</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>316</th>
      <td>Assistant/ Principal Engineer (Electronics/Ele...</td>
      <td>Innosparks Pte Ltd Date Posted: 3-Dec-2017</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>317</th>
      <td>System Engineer (AME/EDC/PMC/JC1)</td>
      <td>Advanced Material Engineering Pte Ltd (AME) Da...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>318</th>
      <td>Innovator</td>
      <td>Ramco Systems Pte Ltd Date Posted: 1-Dec-2017</td>
      <td>[&lt;p&gt;&lt;strong&gt;&lt;span&gt;Job Requirements:&lt;/span&gt;&lt;/st...</td>
    </tr>
    <tr>
      <th>319</th>
      <td>Consultant/Senior Consultant, Instructional De...</td>
      <td>Synpulse Singapore Pte Ltd Date Posted: 13-Dec...</td>
      <td>[&lt;p&gt;&lt;span&gt;&lt;/span&gt;&lt;/p&gt;, &lt;p&gt;&lt;span&gt;As a Consultan...</td>
    </tr>
    <tr>
      <th>320</th>
      <td>Supplier Quality Engineer (KDM/SCM/WC3)</td>
      <td>Singapore Technologies Kinetics Ltd Date Poste...</td>
      <td>[&lt;p _ngcontent-c2=""&gt;78 Shenton Way, #09-01&lt;/p...</td>
    </tr>
    <tr>
      <th>321</th>
      <td>SQL DATABASE ADMINISTRATOR</td>
      <td>Microcool Technologies Pte Ltd Date Posted: 6-...</td>
      <td>[&lt;p&gt;&lt;strong&gt;Job Description:&lt;/strong&gt;&lt;/p&gt;, &lt;p&gt;...</td>
    </tr>
    <tr>
      <th>322</th>
      <td>BUSINESS PROCESS ANALYST</td>
      <td>Micron Semiconductor Asia Date Posted: 21-Dec-...</td>
      <td>[&lt;p&gt;&lt;/p&gt;, &lt;p&gt;Req. ID: 104642 &lt;/p&gt;, &lt;p&gt;As a Bus...</td>
    </tr>
    <tr>
      <th>323</th>
      <td>MSB - Equipment Engineer</td>
      <td>Micron Semiconductor Asia Date Posted: 19-Dec-...</td>
      <td>[&lt;p&gt;&lt;/p&gt;, &lt;p&gt;Req. ID: 104064 &lt;/p&gt;, &lt;p&gt;Responsi...</td>
    </tr>
    <tr>
      <th>324</th>
      <td>NVE SSD Senior/Product Engineer</td>
      <td>Micron Semiconductor Asia Date Posted: 9-Dec-2017</td>
      <td>[&lt;p&gt;&lt;/p&gt;, &lt;p&gt;Req. ID: 99482 &lt;/p&gt;, &lt;p&gt;&lt;strong&gt;R...</td>
    </tr>
    <tr>
      <th>325</th>
      <td>Research Executive</td>
      <td>Majestic Research Services Asia Pte. Limited D...</td>
      <td>[&lt;p&gt;&lt;br/&gt;&lt;/p&gt;, &lt;p&gt;&lt;/p&gt;, &lt;p&gt;&lt;strong&gt;Primary res...</td>
    </tr>
    <tr>
      <th>326</th>
      <td>Project Manager</td>
      <td>Hashmicro Pte Ltd Date Posted: 16-Jan-2018</td>
      <td>[&lt;p&gt;&lt;strong&gt;&lt;span&gt;Job Description&lt;/span&gt;&lt;/stro...</td>
    </tr>
    <tr>
      <th>327</th>
      <td>Product Specialist</td>
      <td>Hashmicro Pte Ltd Date Posted: 16-Jan-2018</td>
      <td>[&lt;p&gt;&lt;br/&gt;&lt;/p&gt;, &lt;p&gt;&lt;strong&gt;Job Description&lt;/str...</td>
    </tr>
  </tbody>
</table>
<p>328 rows × 3 columns</p>
</div>




```python
import sys
sys.setrecursionlimit(100000)
```


```python
pickle_out = open("jobscent","wb")
pickle.dump(df, pickle_out)
pickle_out.close()
```

<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 2 Analysing factors that affect salary, build a regression
### 2.1 Generate meaningful features from text using term frequency–inverse document frequency (TFIDF)

Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query.  

Source: [http://www.tfidf.com/](http://www.tfidf.com/)


```python
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(stop_words='english', ngram_range=(2,3),max_features=2000)
textvec = vect.fit_transform(X['full_description'])
```

__Example of features after tf-idf__


```python
X1 = pd.concat([X, textdf], axis=1, join_axes=[X.index])
X1.head(2)
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
      <th>jobcat</th>
      <th>is_category_dont_care</th>
      <th>is_category_engineer</th>
      <th>is_category_intern</th>
      <th>is_category_leadership</th>
      <th>is_category_scientist</th>
      <th>000 000</th>
      <th>000 000 basic</th>
      <th>000 500</th>
      <th>000 500 higher</th>
      <th>000 basic</th>
      <th>000 basic commensurate</th>
      <th>000 higher</th>
      <th>000 salary</th>
      <th>000 salary commensurate</th>
      <th>00pm salary</th>
      <th>01 singapore</th>
      <th>01c4394 rcb</th>
      <th>01c4394 rcb 200007268e</th>
      <th>01c4394cei registration</th>
      <th>01c4394cei registration r1219269</th>
      <th>04c4294ea personnel</th>
      <th>10 000</th>
      <th>11 000</th>
      <th>1449 confidential</th>
      <th>1449 confidential discussion</th>
      <th>18 01</th>
      <th>18 01 singapore</th>
      <th>200007268e jin</th>
      <th>200007268e jin vatenkeist</th>
      <th>2012 pdpa</th>
      <th>2012 pdpa read</th>
      <th>21 cfr</th>
      <th>21 cfr 820</th>
      <th>30am 00pm</th>
      <th>30am 30pm</th>
      <th>30am 30pm salary</th>
      <th>30am 30pm years</th>
      <th>30am 30pmsalary</th>
      <th>30am 6pm</th>
      <th>30pm salary</th>
      <th>30pm salary 000</th>
      <th>30pm years</th>
      <th>30pm years relevant</th>
      <th>3rd party</th>
      <th>500 500</th>
      <th>500 500 basic</th>
      <th>500 basic</th>
      <th>500 basic negotiable</th>
      <th>500 higher</th>
      <th>500 higher salary</th>
      <th>5k higher</th>
      <th>5k negotiable</th>
      <th>5k negotiable salary</th>
      <th>6536 7890http</th>
      <th>6536 7890http www</th>
      <th>6590 9877</th>
      <th>6590 9877 confidential</th>
      <th>6590 9910</th>
      <th>6590 9910 discussion</th>
      <th>6590 9926</th>
      <th>6590 9926 9230</th>
      <th>6590 9946</th>
      <th>65909963 94783345</th>
      <th>65909963 94783345 confidential</th>
      <th>7890http www</th>
      <th>7890http www peopleprofilers</th>
      <th>8d methodology</th>
      <th>9230 1449</th>
      <th>9230 1449 confidential</th>
      <th>94783345 confidential</th>
      <th>94783345 confidential discussion</th>
      <th>9877 confidential</th>
      <th>9877 confidential discussion</th>
      <th>9910 discussion</th>
      <th>9910 discussion glad</th>
      <th>9926 9230</th>
      <th>9926 9230 1449</th>
      <th>ability communicate</th>
      <th>ability work</th>
      <th>able changeprovide</th>
      <th>able changeprovide timely</th>
      <th>able work</th>
      <th>access data</th>
      <th>access data required</th>
      <th>accordance personal</th>
      <th>accordance personal data</th>
      <th>accordance privacy</th>
      <th>accordance privacy policy</th>
      <th>according experience</th>
      <th>according experience qualification</th>
      <th>according experience qualifications</th>
      <th>account managers</th>
      <th>accounts payable</th>
      <th>achieve career</th>
      <th>achieve employee</th>
      <th>achieve employee satisfaction</th>
      <th>achievegroup asiaor</th>
      <th>achievegroup asiaor friendly</th>
      <th>act 2012</th>
      <th>act 2012 pdpa</th>
      <th>action methods</th>
      <th>action methods statistical</th>
      <th>ad hoc</th>
      <th>ad hoc duties</th>
      <th>added advantage</th>
      <th>address attended</th>
      <th>address attended address</th>
      <th>address job</th>
      <th>address job application</th>
      <th>adelard reg</th>
      <th>adelard reg r1548174</th>
      <th>adhoc duties</th>
      <th>administrative tasks</th>
      <th>advanced excel</th>
      <th>affiliates accordance</th>
      <th>affiliates accordance privacy</th>
      <th>aforementioned address</th>
      <th>aforementioned address attended</th>
      <th>agile ignition</th>
      <th>agile ignition cell</th>
      <th>agile methodologies</th>
      <th>agreed consented</th>
      <th>agreed consented collecting</th>
      <th>agreed terms</th>
      <th>agreed terms privacy</th>
      <th>alternatively send</th>
      <th>alternatively send application</th>
      <th>analysis communication</th>
      <th>analysis communication qms</th>
      <th>analysis data</th>
      <th>analysis data management</th>
      <th>analysis data modellingbasic</th>
      <th>analysis data science</th>
      <th>analysis design</th>
      <th>analysis design generate</th>
      <th>analysis machine</th>
      <th>analysis machine learning</th>
      <th>analysis toolsstrong</th>
      <th>analysis toolsstrong working</th>
      <th>analyst koo</th>
      <th>analyst koo wan</th>
      <th>analyst responsibilities</th>
      <th>analyst responsibilities provide</th>
      <th>analytical problem</th>
      <th>analytical problem solving</th>
      <th>analytical skills</th>
      <th>analytics data</th>
      <th>analytics reporting</th>
      <th>analytics reporting analysis</th>
      <th>analytics strategies</th>
      <th>analytics strategies optimize</th>
      <th>analytics tools</th>
      <th>analyze data</th>
      <th>analyzing datasets</th>
      <th>analyzing datasets excel</th>
      <th>ang kok</th>
      <th>ang kok wee</th>
      <th>ang mo</th>
      <th>applicants apply</th>
      <th>applicants apply sending</th>
      <th>applicants click</th>
      <th>applicants click apply</th>
      <th>applicants kindly</th>
      <th>applicants send</th>
      <th>applicants send resume</th>
      <th>application deemed</th>
      <th>application deemed read</th>
      <th>application email</th>
      <th>application email email</th>
      <th>application emailing</th>
      <th>application emailing detailed</th>
      <th>application employment</th>
      <th>application employment people</th>
      <th>application people</th>
      <th>application people profilers</th>
      <th>application purposes</th>
      <th>application purposes ea</th>
      <th>application sume</th>
      <th>application sume deemed</th>
      <th>applications treated</th>
      <th>applications treated strictest</th>
      <th>apply button</th>
      <th>apply button page</th>
      <th>apply button regret</th>
      <th>apply interested</th>
      <th>apply interested applicants</th>
      <th>apply kindly</th>
      <th>apply sending</th>
      <th>apply sending updated</th>
      <th>apply simply</th>
      <th>apply submit</th>
      <th>apply submit resume</th>
      <th>apply team</th>
      <th>apply team player</th>
      <th>approaches use</th>
      <th>approaches use sound</th>
      <th>approving dhf</th>
      <th>approving dhf validation</th>
      <th>artificial intelligence</th>
      <th>artificial intelligence ai</th>
      <th>ascend4 achievegroup</th>
      <th>ascend4 achievegroup asiaor</th>
      <th>asiaor friendly</th>
      <th>asiaor friendly consultant</th>
      <th>asp net</th>
      <th>assessment correction</th>
      <th>assessment correction containment</th>
      <th>asset management</th>
      <th>assist indicate</th>
      <th>assist indicate information</th>
      <th>attended address</th>
      <th>attended address job</th>
      <th>attractive incentives</th>
      <th>attractive incentives remuneration</th>
      <th>attractive staff</th>
      <th>attractive staff benefits</th>
      <th>automation integration</th>
      <th>automation integration sap</th>
      <th>availability commence</th>
      <th>availability commence work</th>
      <th>availability commence workwe</th>
      <th>availability regret</th>
      <th>availability regret shortlisted</th>
      <th>available corporate</th>
      <th>available corporate website</th>
      <th>available monday</th>
      <th>available monday friday</th>
      <th>aws vb</th>
      <th>bachelor degree</th>
      <th>bachelor degree computer</th>
      <th>bachelor degree engineering</th>
      <th>bachelor masters</th>
      <th>bachelor masters phd</th>
      <th>bachelor science</th>
      <th>bachelor science physics</th>
      <th>bahasa indonesia</th>
      <th>based approaches</th>
      <th>based approaches use</th>
      <th>based experience</th>
      <th>based experience qualification</th>
      <th>based experience qualifications</th>
      <th>basic aws</th>
      <th>basic aws vb</th>
      <th>basic commensurate</th>
      <th>basic commensurate based</th>
      <th>basic negotiable</th>
      <th>basic negotiable higher</th>
      <th>behalf people</th>
      <th>behalf people profilers</th>
      <th>believe make</th>
      <th>believe make difference</th>
      <th>benefits welfare</th>
      <th>benefits welfare training</th>
      <th>best class</th>
      <th>best practices</th>
      <th>big data</th>
      <th>big plus</th>
      <th>black belt</th>
      <th>breakdown expected</th>
      <th>breakdown expected monthly</th>
      <th>build maintain</th>
      <th>business analysis</th>
      <th>business analyst</th>
      <th>business analytics</th>
      <th>business customer</th>
      <th>business data</th>
      <th>business decisions</th>
      <th>business decisions stakeholdersdevelop</th>
      <th>business development</th>
      <th>business engineering</th>
      <th>business engineering data</th>
      <th>business intelligence</th>
      <th>business needs</th>
      <th>business objects</th>
      <th>business objects enterprise</th>
      <th>business process</th>
      <th>business requirements</th>
      <th>business senior</th>
      <th>business technical</th>
      <th>business units</th>
      <th>business users</th>
      <th>business warehouse</th>
      <th>business warehouse products</th>
      <th>business warehouse productshad</th>
      <th>button page</th>
      <th>button page friendly</th>
      <th>button regret</th>
      <th>cad fea</th>
      <th>cad fea cfd</th>
      <th>candidate future</th>
      <th>candidate future suitable</th>
      <th>candidate notified</th>
      <th>candidate notified applications</th>
      <th>candidates expect</th>
      <th>candidates expect competitive</th>
      <th>candidates join</th>
      <th>candidates join growing</th>
      <th>candidates notified</th>
      <th>candidates notified important</th>
      <th>candidates notified submitting</th>
      <th>candidates position</th>
      <th>capa risk</th>
      <th>capa risk assessment</th>
      <th>capability global</th>
      <th>capability global reporting</th>
      <th>career progression</th>
      <th>cash cheque</th>
      <th>cash cheque collection</th>
      <th>cell fusion</th>
      <th>cell fusion automation</th>
      <th>center cic</th>
      <th>center cic ensuring</th>
      <th>cfd software</th>
      <th>cfr 820</th>
      <th>chain crm</th>
      <th>chain crm financeknowledge</th>
      <th>changeprovide timely</th>
      <th>changeprovide timely access</th>
      <th>cheque collection</th>
      <th>chris ng</th>
      <th>cic ensuring</th>
      <th>cic ensuring visualization</th>
      <th>click apply</th>
      <th>click apply button</th>
      <th>click apply submit</th>
      <th>client established</th>
      <th>client global</th>
      <th>client known</th>
      <th>client known established</th>
      <th>client leading</th>
      <th>client leading global</th>
      <th>client leading mnc</th>
      <th>client world</th>
      <th>clients including</th>
      <th>clients including identifying</th>
      <th>clients manage</th>
      <th>clients manage application</th>
      <th>collected used</th>
      <th>collected used disclosed</th>
      <th>collecting using</th>
      <th>collecting using retaining</th>
      <th>collection analysis</th>
      <th>collection analysis communication</th>
      <th>collection systems</th>
      <th>collection systems data</th>
      <th>collection use</th>
      <th>collection use disclosure</th>
      <th>com cn</th>
      <th>com sg</th>
      <th>com sg copy</th>
      <th>com sg privacy</th>
      <th>combined information</th>
      <th>combined information center</th>
      <th>commence work</th>
      <th>commence work regret</th>
      <th>commence workby</th>
      <th>commence workby submitting</th>
      <th>commence workwe</th>
      <th>commence workwe regret</th>
      <th>commensurate according</th>
      <th>commensurate according experience</th>
      <th>commensurate based</th>
      <th>commensurate based experience</th>
      <th>commensurate qualifications</th>
      <th>commensurate qualifications experience</th>
      <th>committed safeguarding</th>
      <th>committed safeguarding personal</th>
      <th>communication interpersonal</th>
      <th>communication interpersonal skills</th>
      <th>communication presentation</th>
      <th>communication qms</th>
      <th>communication qms performance</th>
      <th>communication response</th>
      <th>communication response identified</th>
      <th>communication skills</th>
      <th>company specialised</th>
      <th>company specialised semiconductor</th>
      <th>company strives</th>
      <th>company strives achieve</th>
      <th>company transport</th>
      <th>company transportation</th>
      <th>company transportation pickup</th>
      <th>company transportation provided</th>
      <th>competitive remuneration</th>
      <th>competitive remuneration package</th>
      <th>complex data</th>
      <th>comprehensive range</th>
      <th>comprehensive range benefits</th>
      <th>computer engineering</th>
      <th>computer science</th>
      <th>computer science engineering</th>
      <th>computer science information</th>
      <th>computer science related</th>
      <th>concepts toolsknowledge</th>
      <th>concepts toolsknowledge visual</th>
      <th>conducive working</th>
      <th>conducive working environment</th>
      <th>confidence submitting</th>
      <th>confidence submitting application</th>
      <th>confidence success</th>
      <th>confidence success achievement</th>
      <th>confidential discussion</th>
      <th>confidential discussion indicate</th>
      <th>configuration programming</th>
      <th>configuration programming including</th>
      <th>connection job</th>
      <th>connection job application</th>
      <th>consent drop</th>
      <th>consent drop email</th>
      <th>consented collecting</th>
      <th>consented collecting using</th>
      <th>consented collection</th>
      <th>consented collection use</th>
      <th>consideration regret</th>
      <th>consideration regret short</th>
      <th>consideration success</th>
      <th>consideration success achievement</th>
      <th>consultant 65909963</th>
      <th>consultant 65909963 94783345</th>
      <th>consultant michelle</th>
      <th>consultant michelle 6590</th>
      <th>consultant vivien</th>
      <th>consultant vivien 6590</th>
      <th>consultant wynn</th>
      <th>consultant wynn 6590</th>
      <th>consulting manager</th>
      <th>consumer electronics</th>
      <th>containment communication</th>
      <th>content marketing</th>
      <th>continuous improvement</th>
      <th>control software</th>
      <th>conversion rate</th>
      <th>conversion rate optimization</th>
      <th>copy privacy</th>
      <th>copy privacy policy</th>
      <th>copy resume</th>
      <th>copy resume email</th>
      <th>copy updated</th>
      <th>copy updated resume</th>
      <th>corporate website</th>
      <th>corporate website http</th>
      <th>correction containment</th>
      <th>correction containment communication</th>
      <th>corrective action</th>
      <th>corrective action methods</th>
      <th>corrective preventive</th>
      <th>crm financeknowledge</th>
      <th>crm financeknowledge statistics</th>
      <th>cross functional</th>
      <th>crystal reports</th>
      <th>crystal reports predictive</th>
      <th>current drawn</th>
      <th>current drawn monthly</th>
      <th>current expected</th>
      <th>current expected salary</th>
      <th>current expected salaryreason</th>
      <th>curriculum vitae</th>
      <th>curriculum vitae personal</th>
      <th>custom reports</th>
      <th>customer acquisition</th>
      <th>customer experience</th>
      <th>customer quality</th>
      <th>customer regulatory</th>
      <th>customer regulatory requirementsserve</th>
      <th>customer requirement</th>
      <th>customer satisfaction</th>
      <th>customer service</th>
      <th>customerstransform data</th>
      <th>customerstransform data information</th>
      <th>cutting edge</th>
      <th>cycle management</th>
      <th>cycle management plm</th>
      <th>cycle time</th>
      <th>daily weekly</th>
      <th>dashboards crystal</th>
      <th>dashboards crystal reports</th>
      <th>dashboards scada</th>
      <th>dashboards scada mes</th>
      <th>data accordance</th>
      <th>data accordance personal</th>
      <th>data affiliates</th>
      <th>data affiliates accordance</th>
      <th>data analysis</th>
      <th>data analysis data</th>
      <th>data analyst</th>
      <th>data analyst koo</th>
      <th>data analyst responsibilities</th>
      <th>data analysts</th>
      <th>data analytics</th>
      <th>data analytics strategies</th>
      <th>data architecture</th>
      <th>data collection</th>
      <th>data collection systems</th>
      <th>data connection</th>
      <th>data connection job</th>
      <th>data data</th>
      <th>data driven</th>
      <th>data engineer</th>
      <th>...</th>
      <th>requirements degree</th>
      <th>requirements diploma</th>
      <th>requirements min</th>
      <th>requirements min diploma</th>
      <th>requirements minimum</th>
      <th>requirements years</th>
      <th>requirementsserve qa</th>
      <th>requirementsserve qa ra</th>
      <th>research analysis</th>
      <th>resolve issues</th>
      <th>resolve issues existing</th>
      <th>response identified</th>
      <th>response identified performance</th>
      <th>responsibilities design</th>
      <th>responsibilities develop</th>
      <th>responsibilities ensure</th>
      <th>responsibilities handle</th>
      <th>responsibilities perform</th>
      <th>responsibilities provide</th>
      <th>responsibilities provide support</th>
      <th>responsibilities responsible</th>
      <th>responsibilities work</th>
      <th>resume alternatively</th>
      <th>resume alternatively send</th>
      <th>resume current</th>
      <th>resume current expected</th>
      <th>resume email</th>
      <th>resume email protected</th>
      <th>resume microsoft</th>
      <th>resume microsoft word</th>
      <th>resume microsoft words</th>
      <th>resume ms</th>
      <th>resume ms word</th>
      <th>resume providing</th>
      <th>resume providing details</th>
      <th>resume reason</th>
      <th>resume recent</th>
      <th>resume recent photo</th>
      <th>resumes personal</th>
      <th>resumes personal particulars</th>
      <th>retaining disclosing</th>
      <th>retaining disclosing personal</th>
      <th>risk assessment</th>
      <th>risk assessment correction</th>
      <th>risk based</th>
      <th>risk based approaches</th>
      <th>risk management</th>
      <th>root causes</th>
      <th>safeguarding personal</th>
      <th>safeguarding personal data</th>
      <th>salary 000</th>
      <th>salary 000 000</th>
      <th>salary 3000</th>
      <th>salary 500</th>
      <th>salary commensurate</th>
      <th>salary commensurate according</th>
      <th>salary notice</th>
      <th>salary provide</th>
      <th>salary provide breakdown</th>
      <th>salary reason</th>
      <th>salary reason leaving</th>
      <th>salary4 availability</th>
      <th>salary4 availability regret</th>
      <th>salaryreason leaving</th>
      <th>salaryreason leaving notice</th>
      <th>salaryreason leavingavailability</th>
      <th>salaryreason leavingavailability commence</th>
      <th>sales experience</th>
      <th>sales marketing</th>
      <th>sales strategy</th>
      <th>sales team</th>
      <th>sap business</th>
      <th>sap business warehouse</th>
      <th>sap query</th>
      <th>sap query development</th>
      <th>sas possess</th>
      <th>sas possess knowledge</th>
      <th>satisfaction provides</th>
      <th>satisfaction provides conducive</th>
      <th>satisfaction providing</th>
      <th>satisfaction providing attractive</th>
      <th>satisfy customerstransform</th>
      <th>satisfy customerstransform data</th>
      <th>scada mes</th>
      <th>scada mes information</th>
      <th>science big</th>
      <th>science big data</th>
      <th>science data</th>
      <th>science engineering</th>
      <th>science information</th>
      <th>science information management</th>
      <th>science information technology</th>
      <th>science physics</th>
      <th>science physics mathematics</th>
      <th>science related</th>
      <th>science statistics</th>
      <th>search engine</th>
      <th>self driven</th>
      <th>self motivated</th>
      <th>semiconductor equipments</th>
      <th>semiconductor equipments expansion</th>
      <th>send application</th>
      <th>send application email</th>
      <th>send resume</th>
      <th>send resume microsoft</th>
      <th>send updated</th>
      <th>send updated resume</th>
      <th>sending updated</th>
      <th>sending updated sume</th>
      <th>seng kang</th>
      <th>seng kang woodlands</th>
      <th>sengkang woodlands</th>
      <th>sengkang woodlands working</th>
      <th>senior functional</th>
      <th>senior functional engineer</th>
      <th>sent aforementioned</th>
      <th>sent aforementioned address</th>
      <th>seo sem</th>
      <th>server sap</th>
      <th>server sap business</th>
      <th>servers sap</th>
      <th>servers sap business</th>
      <th>services pte</th>
      <th>services pte committed</th>
      <th>services pte ltdea</th>
      <th>services singapore</th>
      <th>services singapore pte</th>
      <th>sg copy</th>
      <th>sg copy privacy</th>
      <th>sg privacy</th>
      <th>sg privacy php</th>
      <th>shift work</th>
      <th>short listed</th>
      <th>short listed candidate</th>
      <th>shortlisted candidate</th>
      <th>shortlisted candidate notified</th>
      <th>shortlisted candidates</th>
      <th>shortlisted candidates notified</th>
      <th>simply submit</th>
      <th>simply submit application</th>
      <th>singapore pte</th>
      <th>singapore pte ea</th>
      <th>singaporeans information</th>
      <th>singaporeans information location</th>
      <th>skills ability</th>
      <th>skills able</th>
      <th>skills data</th>
      <th>skills experience</th>
      <th>skills knowledge</th>
      <th>skills strong</th>
      <th>social media</th>
      <th>software design</th>
      <th>software development</th>
      <th>software development life</th>
      <th>software engineering</th>
      <th>software tools</th>
      <th>solution design</th>
      <th>solutionsprovide level</th>
      <th>solutionsprovide level support</th>
      <th>solving skills</th>
      <th>sound business</th>
      <th>sound business decisions</th>
      <th>sound investigation</th>
      <th>sound investigation corrective</th>
      <th>south east</th>
      <th>south east asia</th>
      <th>southeast asia</th>
      <th>speaking clients</th>
      <th>specialised semiconductor</th>
      <th>specialised semiconductor equipments</th>
      <th>specifications configuration</th>
      <th>specifications configuration programming</th>
      <th>spoken written</th>
      <th>spss sas</th>
      <th>spss sas possess</th>
      <th>sql nosql</th>
      <th>sql server</th>
      <th>sql server sap</th>
      <th>sql servers</th>
      <th>sql servers sap</th>
      <th>staff apply</th>
      <th>staff apply team</th>
      <th>staff benefits</th>
      <th>staff benefits welfare</th>
      <th>stafflink com</th>
      <th>stafflink com sg</th>
      <th>stafflink services</th>
      <th>stafflink services pte</th>
      <th>staffs apply</th>
      <th>staffs apply team</th>
      <th>stakeholdersdevelop new</th>
      <th>stakeholdersdevelop new databases</th>
      <th>state art</th>
      <th>statement available</th>
      <th>statement available corporate</th>
      <th>statistical analysis</th>
      <th>statistical efficiency</th>
      <th>statistical efficiency quality</th>
      <th>statistical methods</th>
      <th>statistical packages</th>
      <th>statistical packages analyzing</th>
      <th>statistical techniques</th>
      <th>statistics experience</th>
      <th>statistics experience using</th>
      <th>statistics mathematics</th>
      <th>statistics4 years</th>
      <th>statistics4 years relevant</th>
      <th>statisticsknowledge sql</th>
      <th>statisticsknowledge sql server</th>
      <th>strategies optimize</th>
      <th>strategies optimize statistical</th>
      <th>strictest confidence</th>
      <th>strictest confidence submitting</th>
      <th>strictest confidence success</th>
      <th>strives achieve</th>
      <th>strives achieve employee</th>
      <th>strong analytical</th>
      <th>strong analytical skills</th>
      <th>strong experience</th>
      <th>strong knowledge</th>
      <th>strong programming</th>
      <th>strong team</th>
      <th>strong understanding</th>
      <th>strong understanding database</th>
      <th>studio information</th>
      <th>studio information design</th>
      <th>subject matter</th>
      <th>subject matter expert</th>
      <th>subject providing</th>
      <th>subject providing details</th>
      <th>subject title</th>
      <th>subject title systems</th>
      <th>submit application</th>
      <th>submit application emailing</th>
      <th>submit resume</th>
      <th>submit resume providing</th>
      <th>submit updated</th>
      <th>submit updated resume</th>
      <th>submitting application</th>
      <th>submitting application sume</th>
      <th>submitting curriculum</th>
      <th>submitting curriculum vitae</th>
      <th>success achievement</th>
      <th>successful candidates</th>
      <th>successful candidates expect</th>
      <th>successfully maximizing</th>
      <th>successfully maximizing fiscal</th>
      <th>suitability eligibility</th>
      <th>suitability eligibility qualifications</th>
      <th>suitable candidates</th>
      <th>suitable candidates join</th>
      <th>suitable positions</th>
      <th>suitable positions notifying</th>
      <th>suitably qualified</th>
      <th>suitably qualified candidates</th>
      <th>sume deemed</th>
      <th>sume deemed agreed</th>
      <th>sume ms</th>
      <th>sume ms word</th>
      <th>supply chain</th>
      <th>supply chain crm</th>
      <th>support business</th>
      <th>support combined</th>
      <th>support combined information</th>
      <th>support internal</th>
      <th>support troubleshoot</th>
      <th>support troubleshoot resolve</th>
      <th>support visual</th>
      <th>support visual design</th>
      <th>supported operations</th>
      <th>supported operations manufacturing</th>
      <th>systems data</th>
      <th>systems data analytics</th>
      <th>systems engineering</th>
      <th>systems engineering data</th>
      <th>systems experience</th>
      <th>systemsperform analysis</th>
      <th>systemsperform analysis design</th>
      <th>systemsprovide support</th>
      <th>systemsprovide support visual</th>
      <th>tampines seng</th>
      <th>tampines seng kang</th>
      <th>tampines sengkang</th>
      <th>tampines sengkang woodlands</th>
      <th>team build</th>
      <th>team members</th>
      <th>team player</th>
      <th>team player meticulous</th>
      <th>team qms</th>
      <th>team qms quality</th>
      <th>technical issues</th>
      <th>technical specifications</th>
      <th>technical specifications configuration</th>
      <th>technical support</th>
      <th>technical teams</th>
      <th>teng tong</th>
      <th>teng tong lin</th>
      <th>terms privacy</th>
      <th>terms privacy policy</th>
      <th>testing systemsprovide</th>
      <th>testing systemsprovide support</th>
      <th>tham guo</th>
      <th>tham guo yao</th>
      <th>tham ying</th>
      <th>tham ying wen</th>
      <th>thorough understanding</th>
      <th>time management</th>
      <th>timely access</th>
      <th>timely access data</th>
      <th>timely manner</th>
      <th>ting vivien</th>
      <th>ting vivien ea</th>
      <th>title systems</th>
      <th>title systems engineering</th>
      <th>tong lin</th>
      <th>tong lin adelard</th>
      <th>tool web</th>
      <th>tool web intelligence</th>
      <th>tools capability</th>
      <th>tools capability global</th>
      <th>toolsknowledge visual</th>
      <th>toolsknowledge visual agile</th>
      <th>toolsstrong working</th>
      <th>toolsstrong working knowledge</th>
      <th>track record</th>
      <th>training development</th>
      <th>training development opportunities</th>
      <th>training programmes</th>
      <th>training programmes staff</th>
      <th>transport allowance</th>
      <th>transport provided</th>
      <th>transportation pickup</th>
      <th>transportation pickup dropoff</th>
      <th>transportation provided</th>
      <th>treated strictest</th>
      <th>treated strictest confidence</th>
      <th>troubleshoot resolve</th>
      <th>troubleshoot resolve issues</th>
      <th>tuas transport</th>
      <th>ui ux</th>
      <th>understanding database</th>
      <th>understanding database concepts</th>
      <th>understanding machine</th>
      <th>understanding machine learning</th>
      <th>unique individual</th>
      <th>unique individual join</th>
      <th>updated copy</th>
      <th>updated copy resume</th>
      <th>updated resume</th>
      <th>updated resume ms</th>
      <th>updated resume recent</th>
      <th>updated sume</th>
      <th>updated sume ms</th>
      <th>use disclosure</th>
      <th>use disclosure personal</th>
      <th>use risk</th>
      <th>use risk based</th>
      <th>use sound</th>
      <th>use sound investigation</th>
      <th>used disclosed</th>
      <th>used disclosed behalf</th>
      <th>user experience</th>
      <th>user requirements</th>
      <th>using retaining</th>
      <th>using retaining disclosing</th>
      <th>using statistical</th>
      <th>using statistical packages</th>
      <th>using statistical techniques</th>
      <th>validation protocols</th>
      <th>validation protocols risk</th>
      <th>vapor manipulation</th>
      <th>variable bonus</th>
      <th>various industries</th>
      <th>vatenkeist ea</th>
      <th>vatenkeist ea personnel</th>
      <th>vb net</th>
      <th>verbal communication</th>
      <th>verbal written</th>
      <th>visit www</th>
      <th>visit www kellyservices</th>
      <th>visual agile</th>
      <th>visual agile ignition</th>
      <th>visual design</th>
      <th>visual design data</th>
      <th>visualization manufacturing</th>
      <th>visualization manufacturing processes</th>
      <th>vitae personal</th>
      <th>vitae personal data</th>
      <th>vivien 6590</th>
      <th>vivien 6590 9877</th>
      <th>vivien ea</th>
      <th>vivien ea personnel</th>
      <th>walk mrt</th>
      <th>wan ting</th>
      <th>wan ting vivien</th>
      <th>warehouse products</th>
      <th>warehouse products generate</th>
      <th>warehouse productshad</th>
      <th>warehouse productshad experience</th>
      <th>way job</th>
      <th>way job application</th>
      <th>web analytics</th>
      <th>web application</th>
      <th>web development</th>
      <th>web intelligence</th>
      <th>web intelligence dashboards</th>
      <th>web services</th>
      <th>website http</th>
      <th>website http www</th>
      <th>wee gordon</th>
      <th>weekly monthly</th>
      <th>welfare training</th>
      <th>welfare training programmes</th>
      <th>wen ea</th>
      <th>wen ea personnel</th>
      <th>whatsapp tham</th>
      <th>whatsapp tham ying</th>
      <th>wish withdraw</th>
      <th>wish withdraw consent</th>
      <th>withdraw consent</th>
      <th>withdraw consent drop</th>
      <th>woodlands company</th>
      <th>woodlands company transportation</th>
      <th>woodlands good</th>
      <th>woodlands good training</th>
      <th>woodlands working</th>
      <th>woodlands working days</th>
      <th>word format</th>
      <th>word format email</th>
      <th>word format kelly</th>
      <th>word format koo</th>
      <th>word format michelle</th>
      <th>word format subject</th>
      <th>word format teng</th>
      <th>word format wynn</th>
      <th>work closely</th>
      <th>work experience</th>
      <th>work fast</th>
      <th>work fast paced</th>
      <th>work independently</th>
      <th>work mon</th>
      <th>work mon fri</th>
      <th>work regret</th>
      <th>work regret short</th>
      <th>work regret shortlisted</th>
      <th>work team</th>
      <th>work week</th>
      <th>workby submitting</th>
      <th>workby submitting application</th>
      <th>working closely</th>
      <th>working days</th>
      <th>working days days</th>
      <th>working days mon</th>
      <th>working days monday</th>
      <th>working environment</th>
      <th>working environment attractive</th>
      <th>working environment staffs</th>
      <th>working experience</th>
      <th>working hours</th>
      <th>working knowledge</th>
      <th>working knowledge processes</th>
      <th>working location</th>
      <th>workwe regret</th>
      <th>workwe regret short</th>
      <th>world class</th>
      <th>world leading</th>
      <th>written communication</th>
      <th>written spoken</th>
      <th>written verbal</th>
      <th>written verbal communication</th>
      <th>www kellyservices</th>
      <th>www kellyservices com</th>
      <th>www peopleprofilers</th>
      <th>www peopleprofilers comea</th>
      <th>www stafflink</th>
      <th>www stafflink com</th>
      <th>wynn 6590</th>
      <th>wynn 6590 9946</th>
      <th>wynn tham</th>
      <th>wynn tham guo</th>
      <th>yao ea</th>
      <th>yao ea personnel</th>
      <th>year contract</th>
      <th>year experience</th>
      <th>year relevant</th>
      <th>year working</th>
      <th>year working experience</th>
      <th>years experience</th>
      <th>years experience data</th>
      <th>years related</th>
      <th>years relevant</th>
      <th>years relevant experience</th>
      <th>years relevant experiencebachelor</th>
      <th>years relevant experiencepossess</th>
      <th>years relevant working</th>
      <th>years working</th>
      <th>years working experience</th>
      <th>ying wen</th>
      <th>ying wen ea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.120263</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.15104</td>
      <td>0.129234</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.177712</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.170549</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.092119</td>
      <td>0.141244</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.078745</td>
      <td>0.122726</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.11796</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.133566</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.138497</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.139844</td>
      <td>0.170549</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.186683</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.145807</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.133566</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.099726</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.171684</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.211438</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.285001</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.422877</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.247229</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.251267</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.19002</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 2006 columns</p>
</div>



### 2.2 Principal Component Analysis (PCA)  

tf-idf often produces large amounts of features. We may make use of dimensionality reduction techniques like Principal Component Analysis (PCA) to reduce the number of features needed for prediction or classification.


```python
from sklearn.decomposition import PCA
pca = PCA(n_components=15)
pca.fit(X1)
```




    PCA(copy=True, iterated_power='auto', n_components=15, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)



Explained variance shows how much variance in the dataset is accounted for in your principal components.


```python
print(pca.explained_variance_ratio_)  
print(sum(pca.explained_variance_ratio_))
```

    [0.17547903 0.1269256  0.08849329 0.05611374 0.03669752 0.02350112
     0.01906956 0.01444719 0.01286519 0.01149481 0.01019614 0.0093229
     0.00823705 0.00807786 0.00669269]
    0.607613709219762
    

### 2.3 Linear regression to predict salary (with lasso regularisation)


```python
lasso = linear_model.Lasso()
lasso.fit(X_train, y_train)
```




    Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)




```python
cross_val_score(lasso, X_train, y_train)
```




    array([0.33293736, 0.35060433, 0.27800117])




```python
y_pred = lasso.predict(X_test)

# The coefficients
print('Coefficients: \n', lasso.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
```

    Coefficients: 
     [ 1240.71748822  -462.59215024  2215.14691749  -225.41913544
     -2109.89030805  -327.22110887  -976.31850488   608.75956347
      -173.8406491   -525.20852812   625.66194878 -2623.4846474
     -3067.27596946 -1788.03523016  3521.19561575]
    Mean squared error: 2702897.93
    

<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 3. Classifying data science and non-data science jobs
### 3.1 Generate value counts of job categories


```python
all_jobsraw.is_category.value_counts()
```




    engineer      818
    analyst       688
    leadership    624
    dont_care     375
    scientist     361
    intern         54
    database        1
    Name: is_category, dtype: int64



Binarise job categories into data science or non-data science.


```python
all_jobsraw['is_datasci'] = all_jobsraw['is_category'].map(lambda x: 1 if x == 'scientist' else 0)

```

### 3.2 Generate tf-idf matrix from job description


```python
df_text1 = pd.DataFrame(data=df_text.todense(), columns=vect.get_feature_names())
df_text1.head(2)
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
      <th>01 singapore</th>
      <th>01 singapore 079120</th>
      <th>01 singapore 079120 tel</th>
      <th>01 singapore sg</th>
      <th>079120 tel</th>
      <th>079120 tel 6778</th>
      <th>079120 tel 6778 5288</th>
      <th>08c2893 rcb</th>
      <th>09 01</th>
      <th>09 01 singapore</th>
      <th>09 01 singapore 079120</th>
      <th>10 years</th>
      <th>208 368</th>
      <th>208 368 4748</th>
      <th>208 368 4748 keywords</th>
      <th>336 8918</th>
      <th>336 8918 208</th>
      <th>336 8918 208 368</th>
      <th>368 4748</th>
      <th>368 4748 keywords</th>
      <th>368 4748 keywords singapore</th>
      <th>4748 keywords</th>
      <th>4748 keywords singapore</th>
      <th>4748 keywords singapore central</th>
      <th>5288 fax</th>
      <th>5288 fax 6578</th>
      <th>5288 fax 6578 7400</th>
      <th>6578 7400</th>
      <th>6778 5288</th>
      <th>6778 5288 fax</th>
      <th>6778 5288 fax 6578</th>
      <th>78 shenton</th>
      <th>78 shenton way</th>
      <th>78 shenton way 09</th>
      <th>800 336</th>
      <th>800 336 8918</th>
      <th>800 336 8918 208</th>
      <th>8918 208</th>
      <th>8918 208 368</th>
      <th>8918 208 368 4748</th>
      <th>ability work</th>
      <th>ability work independently</th>
      <th>able work</th>
      <th>able work independently</th>
      <th>accommodation team</th>
      <th>accommodation team members</th>
      <th>accommodation team members disabilities</th>
      <th>according experience</th>
      <th>ad hoc</th>
      <th>added advantage</th>
      <th>additional selection</th>
      <th>additional selection criteria</th>
      <th>additional selection criteria based</th>
      <th>administration policy</th>
      <th>administration policy administrator</th>
      <th>administration policy administrator monitor</th>
      <th>administrator human</th>
      <th>administrator human resources</th>
      <th>administrator human resources responsible</th>
      <th>administrator monitor</th>
      <th>administrator monitor compliance</th>
      <th>administrator monitor compliance available</th>
      <th>age national</th>
      <th>age national origin</th>
      <th>age national origin disability</th>
      <th>agreed consented</th>
      <th>agreed consented collecting</th>
      <th>agreed consented collecting using</th>
      <th>amended time</th>
      <th>amended time time</th>
      <th>amended time time notice</th>
      <th>analysis data</th>
      <th>analytical problem</th>
      <th>analytical problem solving</th>
      <th>analytical skills</th>
      <th>analytics data</th>
      <th>answer questions</th>
      <th>answer questions eeo</th>
      <th>answer questions eeo matters</th>
      <th>applicants apply</th>
      <th>applicants notified</th>
      <th>applicants send</th>
      <th>application emailing</th>
      <th>application emailing detailed</th>
      <th>application emailing detailed copy</th>
      <th>application employment</th>
      <th>application employment people</th>
      <th>application employment people profilers</th>
      <th>application people</th>
      <th>application people profilers</th>
      <th>application people profilers collected</th>
      <th>application process</th>
      <th>application process contact</th>
      <th>application process contact micron</th>
      <th>application sume</th>
      <th>application sume deemed</th>
      <th>application sume deemed agreed</th>
      <th>applications treated</th>
      <th>applications treated strictest</th>
      <th>applications treated strictest confidence</th>
      <th>apply button</th>
      <th>apply interested</th>
      <th>apply interested applicants</th>
      <th>apply online</th>
      <th>apply team</th>
      <th>apply team player</th>
      <th>apply team player meticulous</th>
      <th>artificial intelligence</th>
      <th>asia pacific</th>
      <th>assistance application</th>
      <th>assistance application process</th>
      <th>assistance application process contact</th>
      <th>availability commence</th>
      <th>available answer</th>
      <th>available answer questions</th>
      <th>available answer questions eeo</th>
      <th>bachelor degree</th>
      <th>bachelor degree computer</th>
      <th>bachelor degree computer science</th>
      <th>bachelor degree post</th>
      <th>bachelor degree post graduate</th>
      <th>based prevailing</th>
      <th>based prevailing recruitment</th>
      <th>based prevailing recruitment policies</th>
      <th>based singapore</th>
      <th>behalf people</th>
      <th>behalf people profilers</th>
      <th>behalf people profilers determine</th>
      <th>beliefs practices</th>
      <th>beliefs practices manager</th>
      <th>beliefs practices manager supervisor</th>
      <th>believe make</th>
      <th>believe make difference</th>
      <th>believe make difference like</th>
      <th>best practices</th>
      <th>big data</th>
      <th>big data analytics</th>
      <th>business acumen</th>
      <th>business analysis</th>
      <th>business analyst</th>
      <th>business analytics</th>
      <th>business development</th>
      <th>business intelligence</th>
      <th>business needs</th>
      <th>business problems</th>
      <th>business process</th>
      <th>business processes</th>
      <th>business requirements</th>
      <th>business unit</th>
      <th>business units</th>
      <th>business users</th>
      <th>candidate future</th>
      <th>candidate future suitable</th>
      <th>candidate future suitable positions</th>
      <th>candidate notified</th>
      <th>candidate notified applications</th>
      <th>candidate notified applications treated</th>
      <th>candidate possess</th>
      <th>candidates expect</th>
      <th>candidates expect competitive</th>
      <th>candidates expect competitive remuneration</th>
      <th>candidates notified</th>
      <th>candidates wish</th>
      <th>candidates wish apply</th>
      <th>capita pte</th>
      <th>carrying policy</th>
      <th>carrying policy eeo</th>
      <th>carrying policy eeo administrator</th>
      <th>central singapore</th>
      <th>central singapore sg</th>
      <th>central singapore sg 01</th>
      <th>change management</th>
      <th>classifications protected</th>
      <th>classifications protected law</th>
      <th>classifications protected law includes</th>
      <th>click apply</th>
      <th>clients including</th>
      <th>clients including identifying</th>
      <th>clients including identifying potential</th>
      <th>clients manage</th>
      <th>clients manage application</th>
      <th>clients manage application employment</th>
      <th>collected used</th>
      <th>collected used disclosed</th>
      <th>collected used disclosed behalf</th>
      <th>collecting using</th>
      <th>collecting using retaining</th>
      <th>collecting using retaining disclosing</th>
      <th>color religion</th>
      <th>color religion sex</th>
      <th>color religion sex age</th>
      <th>com sg</th>
      <th>commensurate according</th>
      <th>communication interpersonal</th>
      <th>communication interpersonal skills</th>
      <th>communication presentation</th>
      <th>communication skills</th>
      <th>communication skills ability</th>
      <th>competitive remuneration</th>
      <th>competitive remuneration package</th>
      <th>competitive remuneration package comprehensive</th>
      <th>compliance available</th>
      <th>compliance available answer</th>
      <th>compliance available answer questions</th>
      <th>comprehensive range</th>
      <th>comprehensive range benefits</th>
      <th>computer engineering</th>
      <th>computer science</th>
      <th>computer science engineering</th>
      <th>computer science information</th>
      <th>computer science related</th>
      <th>conditions employment</th>
      <th>conditions employment regard</th>
      <th>conditions employment regard person</th>
      <th>confidence submitting</th>
      <th>confidence submitting application</th>
      <th>confidence submitting application sume</th>
      <th>confidential discussion</th>
      <th>confidential discussion indicate</th>
      <th>confidential discussion indicate information</th>
      <th>consented collecting</th>
      <th>consented collecting using</th>
      <th>consented collecting using retaining</th>
      <th>consideration success</th>
      <th>consideration success achievement</th>
      <th>contact micron</th>
      <th>contact micron human</th>
      <th>contact micron human resources</th>
      <th>continuous improvement</th>
      <th>copy updated</th>
      <th>copy updated resume</th>
      <th>copy updated resume ms</th>
      <th>criteria based</th>
      <th>criteria based prevailing</th>
      <th>criteria based prevailing recruitment</th>
      <th>criteria exhaustive</th>
      <th>criteria exhaustive star</th>
      <th>criteria exhaustive star include</th>
      <th>cross functional</th>
      <th>cross functional teams</th>
      <th>current expected</th>
      <th>current expected salary</th>
      <th>current expected salaryreason</th>
      <th>customer service</th>
      <th>cutting edge</th>
      <th>data analysis</th>
      <th>data analyst</th>
      <th>data analytics</th>
      <th>data architecture</th>
      <th>data collection</th>
      <th>data driven</th>
      <th>data governance</th>
      <th>data management</th>
      <th>data mining</th>
      <th>data modelling</th>
      <th>data provided</th>
      <th>data provided way</th>
      <th>data provided way job</th>
      <th>data quality</th>
      <th>data science</th>
      <th>data scientist</th>
      <th>data scientists</th>
      <th>data sets</th>
      <th>data sources</th>
      <th>data visualization</th>
      <th>data warehouse</th>
      <th>day day</th>
      <th>days work</th>
      <th>decision making</th>
      <th>deemed agreed</th>
      <th>deemed agreed consented</th>
      <th>deemed agreed consented collecting</th>
      <th>deep learning</th>
      <th>degree business</th>
      <th>degree computer</th>
      <th>degree computer science</th>
      <th>degree diploma</th>
      <th>degree engineering</th>
      <th>degree post</th>
      <th>degree post graduate</th>
      <th>degree post graduate diploma</th>
      <th>demonstrated ability</th>
      <th>department 800</th>
      <th>department 800 336</th>
      <th>department 800 336 8918</th>
      <th>design develop</th>
      <th>design development</th>
      <th>detailed copy</th>
      <th>detailed copy updated</th>
      <th>detailed copy updated resume</th>
      <th>detailed resume</th>
      <th>determine investigate</th>
      <th>determine investigate suitability</th>
      <th>determine investigate suitability eligibility</th>
      <th>develop implement</th>
      <th>develop maintain</th>
      <th>development experience</th>
      <th>development implementation</th>
      <th>development team</th>
      <th>difference like</th>
      <th>difference like hear</th>
      <th>difference like hear simply</th>
      <th>digital marketing</th>
      <th>diploma degree</th>
      <th>diploma professional</th>
      <th>diploma professional degree</th>
      <th>disabilities religious</th>
      <th>disabilities religious beliefs</th>
      <th>disabilities religious beliefs practices</th>
      <th>disability sexual</th>
      <th>disability sexual orientation</th>
      <th>disability sexual orientation gender</th>
      <th>discipline provide</th>
      <th>discipline provide conditions</th>
      <th>discipline provide conditions employment</th>
      <th>disclosed behalf</th>
      <th>disclosed behalf people</th>
      <th>disclosed behalf people profilers</th>
      <th>disclosing personal</th>
      <th>disclosing personal information</th>
      <th>disclosing personal information prospective</th>
      <th>discussion indicate</th>
      <th>discussion indicate information</th>
      <th>discussion indicate information resume</th>
      <th>duties assigned</th>
      <th>duties responsibilities</th>
      <th>ea licence</th>
      <th>ea license</th>
      <th>ea license 08c2893</th>
      <th>ea personnel</th>
      <th>ea personnel reg</th>
      <th>ea personnel registration</th>
      <th>ea registration</th>
      <th>eeo administrator</th>
      <th>eeo administrator human</th>
      <th>eeo administrator human resources</th>
      <th>eeo matters</th>
      <th>eeo matters request</th>
      <th>eeo matters request assistance</th>
      <th>electrical engineering</th>
      <th>eligibility criteria</th>
      <th>eligibility criteria exhaustive</th>
      <th>eligibility criteria exhaustive star</th>
      <th>eligibility qualifications</th>
      <th>eligibility qualifications employment</th>
      <th>eligibility qualifications employment people</th>
      <th>email detailed</th>
      <th>email email</th>
      <th>email email protected</th>
      <th>email protected</th>
      <th>email protected regret</th>
      <th>email resume</th>
      <th>email resume detailed</th>
      <th>emailing detailed</th>
      <th>emailing detailed copy</th>
      <th>emailing detailed copy updated</th>
      <th>employers consideration</th>
      <th>employers consideration success</th>
      <th>employers consideration success achievement</th>
      <th>employment people</th>
      <th>employment people profilers</th>
      <th>employment people profilers clients</th>
      <th>employment regard</th>
      <th>employment regard person</th>
      <th>employment regard person race</th>
      <th>end end</th>
      <th>engineering computer</th>
      <th>engineering related</th>
      <th>ensure data</th>
      <th>excellent communication</th>
      <th>excellent communication skills</th>
      <th>exhaustive star</th>
      <th>exhaustive star include</th>
      <th>exhaustive star include additional</th>
      <th>existing future</th>
      <th>expect competitive</th>
      <th>expect competitive remuneration</th>
      <th>expect competitive remuneration package</th>
      <th>expected salary</th>
      <th>expected salaryreason</th>
      <th>experience business</th>
      <th>experience data</th>
      <th>experience following</th>
      <th>experience qualifications</th>
      <th>experience related</th>
      <th>experience related field</th>
      <th>experience related field required</th>
      <th>experience using</th>
      <th>experience working</th>
      <th>experience years</th>
      <th>experienced regular</th>
      <th>expression pregnancy</th>
      <th>expression pregnancy veteran</th>
      <th>expression pregnancy veteran status</th>
      <th>fast paced</th>
      <th>fast paced environment</th>
      <th>fax 6578</th>
      <th>fax 6578 7400</th>
      <th>field required</th>
      <th>field required position</th>
      <th>financial services</th>
      <th>following areas</th>
      <th>format email</th>
      <th>format email protected</th>
      <th>friendly consultant</th>
      <th>functional teams</th>
      <th>future suitable</th>
      <th>future suitable positions</th>
      <th>future suitable positions notifying</th>
      <th>gender identity</th>
      <th>gender identity expression</th>
      <th>gender identity expression pregnancy</th>
      <th>good communication</th>
      <th>good communication skills</th>
      <th>good interpersonal</th>
      <th>good knowledge</th>
      <th>good understanding</th>
      <th>graduate diploma</th>
      <th>graduate diploma professional</th>
      <th>graduate diploma professional degree</th>
      <th>growing business</th>
      <th>hands experience</th>
      <th>hardware software</th>
      <th>hear simply</th>
      <th>hear simply submit</th>
      <th>hear simply submit application</th>
      <th>high level</th>
      <th>high quality</th>
      <th>highly motivated</th>
      <th>hire train</th>
      <th>hire train promote</th>
      <th>hire train promote discipline</th>
      <th>human resources</th>
      <th>human resources department</th>
      <th>human resources department 800</th>
      <th>human resources responsible</th>
      <th>human resources responsible administration</th>
      <th>identifying potential</th>
      <th>identifying potential candidate</th>
      <th>identifying potential candidate future</th>
      <th>identity expression</th>
      <th>identity expression pregnancy</th>
      <th>identity expression pregnancy veteran</th>
      <th>importantly believe</th>
      <th>importantly believe make</th>
      <th>importantly believe make difference</th>
      <th>include additional</th>
      <th>include additional selection</th>
      <th>include additional selection criteria</th>
      <th>include following</th>
      <th>includes providing</th>
      <th>includes providing reasonable</th>
      <th>includes providing reasonable accommodation</th>
      <th>including identifying</th>
      <th>including identifying potential</th>
      <th>including identifying potential candidate</th>
      <th>indicate information</th>
      <th>indicate information resume</th>
      <th>indicate information resume current</th>
      <th>inform shortlisted</th>
      <th>inform shortlisted candidates</th>
      <th>inform shortlisted candidates notified</th>
      <th>information location</th>
      <th>information management</th>
      <th>information prospective</th>
      <th>information prospective employers</th>
      <th>information prospective employers consideration</th>
      <th>information resume</th>
      <th>information resume current</th>
      <th>information resume current expected</th>
      <th>information security</th>
      <th>information systems</th>
      <th>information technology</th>
      <th>informed personal</th>
      <th>informed personal data</th>
      <th>informed personal data provided</th>
      <th>interested applicants</th>
      <th>interested applicants apply</th>
      <th>interested candidates</th>
      <th>interested candidates wish</th>
      <th>interested candidates wish apply</th>
      <th>internal external</th>
      <th>interpersonal communication</th>
      <th>interpersonal skills</th>
      <th>investigate suitability</th>
      <th>investigate suitability eligibility</th>
      <th>investigate suitability eligibility qualifications</th>
      <th>job application</th>
      <th>job application people</th>
      <th>job application people profilers</th>
      <th>job description</th>
      <th>job id</th>
      <th>job requirements</th>
      <th>job responsibilities</th>
      <th>job segment</th>
      <th>join growing</th>
      <th>key responsibilities</th>
      <th>key stakeholders</th>
      <th>keywords singapore</th>
      <th>...</th>
      <th>kindly send</th>
      <th>knowledge data</th>
      <th>knowledge experience</th>
      <th>language processing</th>
      <th>large scale</th>
      <th>law includes</th>
      <th>law includes providing</th>
      <th>law includes providing reasonable</th>
      <th>learning data</th>
      <th>learning models</th>
      <th>learning techniques</th>
      <th>li sing</th>
      <th>li sing job</th>
      <th>li sing job segment</th>
      <th>licence number</th>
      <th>license 08c2893</th>
      <th>license number</th>
      <th>life cycle</th>
      <th>like hear</th>
      <th>like hear simply</th>
      <th>like hear simply submit</th>
      <th>listed candidate</th>
      <th>listed candidate notified</th>
      <th>listed candidate notified applications</th>
      <th>listed candidates</th>
      <th>long term</th>
      <th>machine learning</th>
      <th>machine learning techniques</th>
      <th>make difference</th>
      <th>make difference like</th>
      <th>make difference like hear</th>
      <th>manage application</th>
      <th>manage application employment</th>
      <th>manage application employment people</th>
      <th>management experience</th>
      <th>management skills</th>
      <th>management team</th>
      <th>manager supervisor</th>
      <th>manager supervisor team</th>
      <th>manager supervisor team member</th>
      <th>manufacturing engineer</th>
      <th>market intelligence</th>
      <th>market research</th>
      <th>master degree</th>
      <th>matters request</th>
      <th>matters request assistance</th>
      <th>matters request assistance application</th>
      <th>member responsible</th>
      <th>member responsible carrying</th>
      <th>member responsible carrying policy</th>
      <th>members disabilities</th>
      <th>members disabilities religious</th>
      <th>members disabilities religious beliefs</th>
      <th>meticulous organized</th>
      <th>meticulous organized importantly</th>
      <th>meticulous organized importantly believe</th>
      <th>micron human</th>
      <th>micron human resources</th>
      <th>micron human resources department</th>
      <th>microsoft excel</th>
      <th>microsoft office</th>
      <th>min years</th>
      <th>minimum years</th>
      <th>minimum years experience</th>
      <th>minimum years relevant</th>
      <th>mon fri</th>
      <th>monday friday</th>
      <th>monitor compliance</th>
      <th>monitor compliance available</th>
      <th>monitor compliance available answer</th>
      <th>ms excel</th>
      <th>ms office</th>
      <th>ms word</th>
      <th>ms word format</th>
      <th>ms word format email</th>
      <th>national origin</th>
      <th>national origin disability</th>
      <th>national origin disability sexual</th>
      <th>natural language</th>
      <th>new business</th>
      <th>new product</th>
      <th>new products</th>
      <th>notice period</th>
      <th>notice period availability</th>
      <th>notice regret</th>
      <th>notice regret shortlisted</th>
      <th>notice regret shortlisted candidates</th>
      <th>notified applications</th>
      <th>notified applications treated</th>
      <th>notified applications treated strictest</th>
      <th>notifying positions</th>
      <th>notifying positions existing</th>
      <th>notifying positions existing future</th>
      <th>organized importantly</th>
      <th>organized importantly believe</th>
      <th>organized importantly believe make</th>
      <th>orientation gender</th>
      <th>orientation gender identity</th>
      <th>orientation gender identity expression</th>
      <th>origin disability</th>
      <th>origin disability sexual</th>
      <th>origin disability sexual orientation</th>
      <th>paced environment</th>
      <th>package comprehensive</th>
      <th>package comprehensive range</th>
      <th>package comprehensive range benefits</th>
      <th>people profilers</th>
      <th>people profilers clients</th>
      <th>people profilers clients including</th>
      <th>people profilers clients manage</th>
      <th>people profilers collected</th>
      <th>people profilers collected used</th>
      <th>people profilers determine</th>
      <th>people profilers determine investigate</th>
      <th>period availability</th>
      <th>period availability commence</th>
      <th>person race</th>
      <th>person race color</th>
      <th>person race color religion</th>
      <th>personal data</th>
      <th>personal data provided</th>
      <th>personal data provided way</th>
      <th>personal information</th>
      <th>personal information prospective</th>
      <th>personal information prospective employers</th>
      <th>personnel reg</th>
      <th>personnel registration</th>
      <th>player meticulous</th>
      <th>player meticulous organized</th>
      <th>player meticulous organized importantly</th>
      <th>policies amended</th>
      <th>policies amended time</th>
      <th>policies amended time time</th>
      <th>policies policies</th>
      <th>policies policies amended</th>
      <th>policies policies amended time</th>
      <th>policy administrator</th>
      <th>policy administrator monitor</th>
      <th>policy administrator monitor compliance</th>
      <th>policy eeo</th>
      <th>policy eeo administrator</th>
      <th>policy eeo administrator human</th>
      <th>positions existing</th>
      <th>positions existing future</th>
      <th>positions notifying</th>
      <th>positions notifying positions</th>
      <th>positions notifying positions existing</th>
      <th>possess strong</th>
      <th>post graduate</th>
      <th>post graduate diploma</th>
      <th>post graduate diploma professional</th>
      <th>potential candidate</th>
      <th>potential candidate future</th>
      <th>potential candidate future suitable</th>
      <th>practices manager</th>
      <th>practices manager supervisor</th>
      <th>practices manager supervisor team</th>
      <th>pre sales</th>
      <th>pregnancy veteran</th>
      <th>pregnancy veteran status</th>
      <th>pregnancy veteran status classifications</th>
      <th>presentation skills</th>
      <th>prevailing recruitment</th>
      <th>prevailing recruitment policies</th>
      <th>prevailing recruitment policies policies</th>
      <th>prior experience</th>
      <th>privacy policy</th>
      <th>problem solving</th>
      <th>problem solving skills</th>
      <th>process contact</th>
      <th>process contact micron</th>
      <th>process contact micron human</th>
      <th>process improvement</th>
      <th>product development</th>
      <th>product management</th>
      <th>products services</th>
      <th>professional degree</th>
      <th>profilers clients</th>
      <th>profilers clients including</th>
      <th>profilers clients including identifying</th>
      <th>profilers clients manage</th>
      <th>profilers clients manage application</th>
      <th>profilers collected</th>
      <th>profilers collected used</th>
      <th>profilers collected used disclosed</th>
      <th>profilers determine</th>
      <th>profilers determine investigate</th>
      <th>profilers determine investigate suitability</th>
      <th>profilers pte</th>
      <th>programming languages</th>
      <th>project management</th>
      <th>project manager</th>
      <th>promote discipline</th>
      <th>promote discipline provide</th>
      <th>promote discipline provide conditions</th>
      <th>prospective employers</th>
      <th>prospective employers consideration</th>
      <th>prospective employers consideration success</th>
      <th>protected law</th>
      <th>protected law includes</th>
      <th>protected law includes providing</th>
      <th>protected regret</th>
      <th>proven track</th>
      <th>proven track record</th>
      <th>provide conditions</th>
      <th>provide conditions employment</th>
      <th>provide conditions employment regard</th>
      <th>provide support</th>
      <th>provide technical</th>
      <th>provided way</th>
      <th>provided way job</th>
      <th>provided way job application</th>
      <th>providing reasonable</th>
      <th>providing reasonable accommodation</th>
      <th>providing reasonable accommodation team</th>
      <th>pte ea</th>
      <th>pte ea license</th>
      <th>qualifications employment</th>
      <th>qualifications employment people</th>
      <th>qualifications employment people profilers</th>
      <th>quality assurance</th>
      <th>questions eeo</th>
      <th>questions eeo matters</th>
      <th>questions eeo matters request</th>
      <th>race color</th>
      <th>race color religion</th>
      <th>race color religion sex</th>
      <th>range benefits</th>
      <th>real estate</th>
      <th>real time</th>
      <th>reason leaving</th>
      <th>reasonable accommodation</th>
      <th>reasonable accommodation team</th>
      <th>reasonable accommodation team members</th>
      <th>recruit hire</th>
      <th>recruit hire train</th>
      <th>recruit hire train promote</th>
      <th>recruitment policies</th>
      <th>recruitment policies policies</th>
      <th>recruitment policies policies amended</th>
      <th>regard person</th>
      <th>regard person race</th>
      <th>regard person race color</th>
      <th>registration number</th>
      <th>regret inform</th>
      <th>regret inform shortlisted</th>
      <th>regret inform shortlisted candidates</th>
      <th>regret short</th>
      <th>regret short listed</th>
      <th>regret short listed candidate</th>
      <th>regret shortlisted</th>
      <th>regret shortlisted candidates</th>
      <th>regret shortlisted candidates notified</th>
      <th>regular engineering</th>
      <th>related discipline</th>
      <th>related field</th>
      <th>related field required</th>
      <th>related field required position</th>
      <th>relevant experience</th>
      <th>relevant working</th>
      <th>relevant working experience</th>
      <th>religion sex</th>
      <th>religion sex age</th>
      <th>religion sex age national</th>
      <th>religious beliefs</th>
      <th>religious beliefs practices</th>
      <th>religious beliefs practices manager</th>
      <th>remuneration package</th>
      <th>remuneration package comprehensive</th>
      <th>remuneration package comprehensive range</th>
      <th>req id</th>
      <th>request assistance</th>
      <th>request assistance application</th>
      <th>request assistance application process</th>
      <th>required position</th>
      <th>requirements bachelor</th>
      <th>requirements bachelor degree</th>
      <th>requirements degree</th>
      <th>requirements diploma</th>
      <th>requirements min</th>
      <th>requirements minimum</th>
      <th>resources department</th>
      <th>resources department 800</th>
      <th>resources department 800 336</th>
      <th>resources responsible</th>
      <th>resources responsible administration</th>
      <th>resources responsible administration policy</th>
      <th>responsibilities include</th>
      <th>responsibilities provide</th>
      <th>responsibilities requirements</th>
      <th>responsible administration</th>
      <th>responsible administration policy</th>
      <th>responsible administration policy administrator</th>
      <th>responsible carrying</th>
      <th>responsible carrying policy</th>
      <th>responsible carrying policy eeo</th>
      <th>resume current</th>
      <th>resume current expected</th>
      <th>resume current expected salaryreason</th>
      <th>resume detailed</th>
      <th>resume email</th>
      <th>resume email protected</th>
      <th>resume ms</th>
      <th>resume ms word</th>
      <th>resume ms word format</th>
      <th>retaining disclosing</th>
      <th>retaining disclosing personal</th>
      <th>retaining disclosing personal information</th>
      <th>risk management</th>
      <th>root cause</th>
      <th>salary commensurate</th>
      <th>sales marketing</th>
      <th>sales team</th>
      <th>science computer</th>
      <th>science engineering</th>
      <th>science information</th>
      <th>science related</th>
      <th>science technology</th>
      <th>selection criteria</th>
      <th>selection criteria based</th>
      <th>selection criteria based prevailing</th>
      <th>self motivated</th>
      <th>send resume</th>
      <th>send resume email</th>
      <th>send resume email protected</th>
      <th>send updated</th>
      <th>senior management</th>
      <th>services pte</th>
      <th>sex age</th>
      <th>sex age national</th>
      <th>sex age national origin</th>
      <th>sexual orientation</th>
      <th>sexual orientation gender</th>
      <th>sexual orientation gender identity</th>
      <th>sg 01</th>
      <th>sg 01 singapore</th>
      <th>sg 01 singapore sg</th>
      <th>shenton way</th>
      <th>shenton way 09</th>
      <th>shenton way 09 01</th>
      <th>short listed</th>
      <th>short listed candidate</th>
      <th>short listed candidate notified</th>
      <th>short listed candidates</th>
      <th>shortlisted applicants</th>
      <th>shortlisted applicants notified</th>
      <th>shortlisted candidates</th>
      <th>shortlisted candidates notified</th>
      <th>simply submit</th>
      <th>simply submit application</th>
      <th>simply submit application emailing</th>
      <th>sing job</th>
      <th>sing job segment</th>
      <th>singapore 079120</th>
      <th>singapore 079120 tel</th>
      <th>singapore 079120 tel 6778</th>
      <th>singapore central</th>
      <th>singapore central singapore</th>
      <th>singapore central singapore sg</th>
      <th>singapore sg</th>
      <th>singapore sg 01</th>
      <th>singapore sg 01 singapore</th>
      <th>skills ability</th>
      <th>skills able</th>
      <th>skills experience</th>
      <th>skills strong</th>
      <th>social media</th>
      <th>software development</th>
      <th>software engineering</th>
      <th>solving skills</th>
      <th>sql server</th>
      <th>star include</th>
      <th>star include additional</th>
      <th>star include additional selection</th>
      <th>state art</th>
      <th>statistical analysis</th>
      <th>status classifications</th>
      <th>status classifications protected</th>
      <th>status classifications protected law</th>
      <th>strictest confidence</th>
      <th>strictest confidence submitting</th>
      <th>strictest confidence submitting application</th>
      <th>strong analytical</th>
      <th>strong analytical skills</th>
      <th>strong communication</th>
      <th>strong knowledge</th>
      <th>strong understanding</th>
      <th>subject matter</th>
      <th>submit application</th>
      <th>submit application emailing</th>
      <th>submit application emailing detailed</th>
      <th>submit resume</th>
      <th>submitting application</th>
      <th>submitting application sume</th>
      <th>submitting application sume deemed</th>
      <th>success achievement</th>
      <th>successful candidate</th>
      <th>successful candidates</th>
      <th>successful candidates expect</th>
      <th>successful candidates expect competitive</th>
      <th>suitability eligibility</th>
      <th>suitability eligibility qualifications</th>
      <th>suitability eligibility qualifications employment</th>
      <th>suitable positions</th>
      <th>suitable positions notifying</th>
      <th>suitable positions notifying positions</th>
      <th>sume deemed</th>
      <th>sume deemed agreed</th>
      <th>sume deemed agreed consented</th>
      <th>supervisor team</th>
      <th>supervisor team member</th>
      <th>supervisor team member responsible</th>
      <th>supply chain</th>
      <th>support business</th>
      <th>team member</th>
      <th>team member responsible</th>
      <th>team member responsible carrying</th>
      <th>team members</th>
      <th>team members disabilities</th>
      <th>team members disabilities religious</th>
      <th>team player</th>
      <th>team player meticulous</th>
      <th>team player meticulous organized</th>
      <th>technical support</th>
      <th>tel 6778</th>
      <th>tel 6778 5288</th>
      <th>tel 6778 5288 fax</th>
      <th>time notice</th>
      <th>time notice regret</th>
      <th>time notice regret shortlisted</th>
      <th>time time</th>
      <th>time time notice</th>
      <th>time time notice regret</th>
      <th>timely manner</th>
      <th>tools like</th>
      <th>track record</th>
      <th>train promote</th>
      <th>train promote discipline</th>
      <th>train promote discipline provide</th>
      <th>treated strictest</th>
      <th>treated strictest confidence</th>
      <th>treated strictest confidence submitting</th>
      <th>understand business</th>
      <th>updated resume</th>
      <th>updated resume ms</th>
      <th>updated resume ms word</th>
      <th>used disclosed</th>
      <th>used disclosed behalf</th>
      <th>used disclosed behalf people</th>
      <th>using data</th>
      <th>using retaining</th>
      <th>using retaining disclosing</th>
      <th>using retaining disclosing personal</th>
      <th>using statistical</th>
      <th>verbal communication</th>
      <th>verbal communication skills</th>
      <th>verbal written</th>
      <th>verbal written communication</th>
      <th>veteran status</th>
      <th>veteran status classifications</th>
      <th>veteran status classifications protected</th>
      <th>visit www</th>
      <th>way 09</th>
      <th>way 09 01</th>
      <th>way 09 01 singapore</th>
      <th>way job</th>
      <th>way job application</th>
      <th>way job application people</th>
      <th>wish apply</th>
      <th>word format</th>
      <th>word format email</th>
      <th>word format email protected</th>
      <th>work closely</th>
      <th>work experience</th>
      <th>work fast</th>
      <th>work independently</th>
      <th>work team</th>
      <th>working closely</th>
      <th>working days</th>
      <th>working environment</th>
      <th>working experience</th>
      <th>working experience related</th>
      <th>working experience related field</th>
      <th>working knowledge</th>
      <th>working location</th>
      <th>world class</th>
      <th>written communication</th>
      <th>written communication skills</th>
      <th>written verbal</th>
      <th>written verbal communication</th>
      <th>year working</th>
      <th>year working experience</th>
      <th>year working experience related</th>
      <th>years experience</th>
      <th>years relevant</th>
      <th>years relevant experience</th>
      <th>years relevant working</th>
      <th>years working</th>
      <th>years working experience</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.173089</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.190167</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.181393</td>
      <td>0.183186</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.196003</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.10019</td>
      <td>0.177626</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.117544</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.16711</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.144109</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.284440</td>
      <td>0.170323</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.181834</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.158506</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.399907</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.125963</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.130361</td>
      <td>0.174546</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.398536</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.171339</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.164131</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.161088</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.190714</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.138685</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.159777</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.106457</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.22879</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.309515</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.560493</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.246489</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.243257</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.246489</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.340835</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.430903</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.234881</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 1000 columns</p>
</div>



### 3.3 Build a logistic regression and random forest classifier to classify the dataset
__Logistic regression__


```python
from sklearn.metrics import classification_report
target_names = ['non-science', 'science']
print(classification_report(y_test, y_pred, target_names=target_names))
```

                 precision    recall  f1-score   support
    
    non-science       0.96      0.94      0.95       770
        science       0.60      0.71      0.65       107
    
    avg / total       0.92      0.91      0.91       877
    
    

__Random forest classifier__


```python
target_names = ['non-science', 'science']
print(classification_report(y_test, y_pred, target_names=target_names))
```

                 precision    recall  f1-score   support
    
    non-science       0.95      0.94      0.95       770
        science       0.60      0.65      0.62       107
    
    avg / total       0.91      0.90      0.91       877
    
    
