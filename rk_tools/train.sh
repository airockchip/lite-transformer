###
 # Copyright (c) 2022 by Rockchip Electronics Co., Ltd. All Rights Reserved.
 # 
 # 
 # @Author: Randall Zhuo
 # @Date: 2022-09-16 11:25:37
 # @LastEditors: Randall
 # @LastEditTime: 2022-09-16 11:26:58
 # @Description: TODO
### 

CUDA_VISIBLE_DEVICES="0,1,2" python3 train.py ~/randall_ssd_share/dataset/ncv15_data --configs configs/rk_nmt.yml 
