git clone https://hf-mirror.com/Salesforce/blip2-opt-2.7b
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_CKE_pretrained_vinvl.tar.gz -O pretrained_vinvl.tar.gz
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_CKE_pretrained_avg.tar.gz -O pretrained_avg_model.tar.gz

tar -xvf pretrained_vinvl.tar.gz
tar -xvf pretrained_avg_model.tar.gz

mkdir pretrained_base
mv pretrained_model/pretrained_base/checkpoint-2000000/* pretrained_base/
rm -r pretrained_model