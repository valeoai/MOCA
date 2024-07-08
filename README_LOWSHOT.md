# Low-shot ImageNet classification evaluation using logistic regression

## Produce the 1-shot results
```bash
DATA_PATH="path_to_ImageNet1k"
PRETRAINED_DIR="moca_vitb16_200epochs_mom994_wd05to20_v1p0"

# 1st set
python eval_lowshot_logistic.py --model_name vit_base_patch16 --preload --subset_path imagenet_subsets1/1imgs_class.txt --device cuda:0 --penalty l2 --nthreads 8 --pretrained "${PRETRAINED_DIR}/checkpoint-last.pth" --output_dir "${PRETRAINED_DIR}/logist_eval/" --data_path "${DATA_PATH}" --lambd 0.1
# Expected results:
# test score (lambd_val 0.1): 0.55226

# 2nd set
python eval_lowshot_logistic.py --model_name vit_base_patch16 --preload --subset_path imagenet_subsets2/1imgs_class.txt --device cuda:0 --penalty l2 --nthreads 8 --pretrained "${PRETRAINED_DIR}/checkpoint-last.pth" --output_dir "${PRETRAINED_DIR}/logist_eval/" --data_path "${DATA_PATH}" --lambd 0.1
# Expected results:
# test score (lambd_val 0.1): 0.554

# 3rd set
python eval_lowshot_logistic.py --model_name vit_base_patch16 --preload --subset_path imagenet_subsets3/1imgs_class.txt --device cuda:0 --penalty l2 --nthreads 8 --pretrained "${PRETRAINED_DIR}/checkpoint-last.pth" --output_dir "${PRETRAINED_DIR}/logist_eval/" --data_path "${DATA_PATH}" --lambd 0.1
# Expected results:
# test score (lambd_val 0.1): 0.55064
```

## Produce the 2-shot results
```bash
# 1st set
python eval_lowshot_logistic.py --model_name vit_base_patch16 --preload --subset_path imagenet_subsets1/2imgs_class.txt --device cuda:0 --penalty l2 --nthreads 8 --pretrained "${PRETRAINED_DIR}/checkpoint-last.pth" --output_dir "${PRETRAINED_DIR}/logist_eval/" --data_path "${DATA_PATH}" --lambd 0.1
# Expected results:
# test score (lambd_val 0.1): 0.63822

# 2nd set
python eval_lowshot_logistic.py --model_name vit_base_patch16 --preload --subset_path imagenet_subsets2/2imgs_class.txt --device cuda:0 --penalty l2 --nthreads 8 --pretrained "${PRETRAINED_DIR}/checkpoint-last.pth" --output_dir "${PRETRAINED_DIR}/logist_eval/" --data_path "${DATA_PATH}" --lambd 0.1
# Expected results:
# test score (lambd_val 0.1): 0.64314

# 3rd set
python eval_lowshot_logistic.py --model_name vit_base_patch16 --preload --subset_path imagenet_subsets3/2imgs_class.txt --device cuda:0 --penalty l2 --nthreads 8 --pretrained "${PRETRAINED_DIR}/checkpoint-last.pth" --output_dir "${PRETRAINED_DIR}/logist_eval/" --data_path "${DATA_PATH}" --lambd 0.1
# Expected results:
# test score (lambd_val 0.1): 0.63692
```

## Produce the 5-shot results
```bash
# 1st set
python eval_lowshot_logistic.py --model_name vit_base_patch16 --preload --subset_path imagenet_subsets1/5imgs_class.txt --device cuda:0 --penalty l2 --nthreads 8 --pretrained "${PRETRAINED_DIR}/checkpoint-last.pth" --output_dir "${PRETRAINED_DIR}/logist_eval/" --data_path "${DATA_PATH}" --lambd 0.1
# Expected results:
# test score (lambd_val 0.1): 0.69898

# 2nd set
python eval_lowshot_logistic.py --model_name vit_base_patch16 --preload --subset_path imagenet_subsets2/5imgs_class.txt --device cuda:0 --penalty l2 --nthreads 8 --pretrained "${PRETRAINED_DIR}/checkpoint-last.pth" --output_dir "${PRETRAINED_DIR}/logist_eval/" --data_path "${DATA_PATH}" --lambd 0.1
# Expected results:
# test score (lambd_val 0.1): 0.69708

# 3rd set
python eval_lowshot_logistic.py --model_name vit_base_patch16 --preload --subset_path imagenet_subsets3/5imgs_class.txt --device cuda:0 --penalty l2 --nthreads 8 --pretrained "${PRETRAINED_DIR}/checkpoint-last.pth" --output_dir "${PRETRAINED_DIR}/logist_eval/" --data_path "${DATA_PATH}" --lambd 0.1
# Expected results:
# test score (lambd_val 0.1): 0.69614
```

## Produce the 1-percent setting results
```bash
python eval_lowshot_logistic.py --model_name vit_base_patch16 --preload --subset_path imagenet_subsets1/1percent.txt --device cuda:0 --penalty l2 --nthreads 8 --pretrained "${PRETRAINED_DIR}/checkpoint-last.pth" --output_dir "${PRETRAINED_DIR}/logist_eval/" --data_path "${DATA_PATH}" --lambd 0.1
# Expected results:
# test score (lambd_val 0.1): 0.72438
```

## Compute mean and std for the 1-shot, 2-shot, and 5-shot results 
```bash
import numpy as np
# For 1 images per class with lambd=0.1000
results = np.asarray([0.55226, 0.554, 0.55064])
(results*100).mean()
# 55.23
(results*100).std()
# 0.13720058308914232

# For 2 images per class with lambd=0.1000
results = np.asarray([0.63822, 0.64314, 0.63692])
(results*100).mean()
# 63.942666666666675
(results*100).std()
# 0.26788222951306273

# For 5 images per class with lambd=0.1000
results = np.asarray([0.69898, 0.69708, 0.69614])
(results*100).mean()
# 69.74
(results*100).std()
# 0.11812987203356917
```
