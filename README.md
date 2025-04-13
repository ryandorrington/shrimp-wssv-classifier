## Setup
1. 
```
git clone https://github.com/ryandorrington/shrimp-wssv-classifier.git
cd shrimp-wssv-classifier
pip install -r requirements.txt
```

2. Downlad the dataset: https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/jhrtdj9txm-3.zip
 
3. Once downloaded:
```
mv ~/Downloads/ShrimpDiseaseImageBD\ An\ Image\ Dataset\ for\ Computer\ Vision-Based\ Detection\ of\ Shrimp\ Diseases\ in\ Bangladesh.zip ./
```

4. unzip dataset and image subdirectories
5. Delete BG data:
```
rm -rf ShrimpDiseaseImageBD\ An\ Image\ Dataset\ for\ Computer\ Vision-Based\ Detection\ of\ Shrimp\ Diseases\ in\ Bangladesh/Root/Raw\ Images/4.\ WSSV_BG  ShrimpDiseaseImageBD\ An\ Image\ Dataset\ for\ Computer\ Vision-Based\ Detection\ of\ Shrimp\ Diseases\ in\ Bangladesh/Root/Raw\ Images/2.\ BG
```





Dataset source: Islam, Mohammad Manzurul; Sarker, Anabil; Choudhury, Ashiquzzaman ; Ahmed, Noortaz ; Rasel, Ahmed Abdal Shafi Rasel (2025), “ShrimpDiseaseImageBD: An Image Dataset for Computer Vision-Based Detection of Shrimp Diseases in Bangladesh”, Mendeley Data, V3, doi: 10.17632/jhrtdj9txm.3
