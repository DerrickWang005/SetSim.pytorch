# Object Detection on VOC 2007+2012
# 24epoch Faster RCNN (C4-backbone)
# for num in {1..5}
# do
# python train_net.py --config-file ./configs/pascal_voc_R_50_C4_24k_base.yaml \
#                     --num-gpus 8 --dist-url tcp://127.0.0.1:1681 \
#                     MODEL.WEIGHTS ./r50_SeF_Sym2x_200ep.pkl \
#                     OUTPUT_DIR ./exp_voc/SemanticFocus_attention0p7_negtive0p2_nearest1_geometry0p5_Sym2x_ep200_trial_$num
# done

# Object Detection & Instance Segmentation on COCO 2017
# 1x schedule Mask RCNN (FPN)
# python train_net.py --config-file configs/coco_R_50_FPN_1x_base.yaml \
#                               --num-gpus 8 --dist-url tcp://127.0.0.1:2681 \
#                               MODEL.WEIGHTS ./detection/output_densecl_200e.pkl \
#                               OUTPUT_DIR ./exp_coco1x/SemanticFocus_attention0p7_negtive0p2_nearest1_geometry0p5_Sym2x_ep200
#                             #   --eval-only --resume \
# 2x schedule Mask RCNN (FPN)
python train_net.py --config-file ./configs/coco_R_50_FPN_2x_base.yaml \
                              --num-gpus 8 --dist-url tcp://127.0.0.1:3681 \
                              MODEL.WEIGHTS ./r50_SeF_Sym2x_200ep.pkl \
                              OUTPUT_DIR ./exp_coco2x/SemanticFocus_attention0p7_negtive0p2_nearest1_geometry0p5_Sym2x_ep200

python train_net.py --config-file ./configs/coco_R_50_FPN_2x_base.yaml \
                                --num-gpus 8 --dist-url tcp://127.0.0.1:1681 \
                                MODEL.WEIGHTS ./pretrain/r50_moco_v1_200ep.pkl \
                                OUTPUT_DIR ./exp_coco2x/moco_v1_200ep
                                
python train_net.py --config-file ./configs/coco_R_50_FPN_2x_base.yaml \
                                --num-gpus 8 --dist-url tcp://127.0.0.1:1681 \
                                MODEL.WEIGHTS ./pretrain/r50_moco_v2_200ep.pkl \
                                OUTPUT_DIR ./exp_coco2x/moco_v2_200ep

python train_net.py --config-file ./configs/coco_R_50_FPN_2x_base.yaml \
                                --num-gpus 8 --dist-url tcp://127.0.0.1:1681 \
                                MODEL.WEIGHTS ./pretrain/r50_densecl_200ep.pkl \
                                OUTPUT_DIR ./exp_coco2x/densecl_200ep

python train_net.py --config-file ./configs/coco_R_50_FPN_2x_base.yaml \
                                --num-gpus 8 --dist-url tcp://127.0.0.1:1681 \
                                MODEL.WEIGHTS ./pretrain/r50_pixpro_100ep.pkl \
                                OUTPUT_DIR ./exp_coco2x/pixpro_100ep

python train_net.py --config-file ./configs/coco_R_50_FPN_2x_base.yaml \
                                --num-gpus 8 --dist-url tcp://127.0.0.1:1681 \
                                MODEL.WEIGHTS ./pretrain/R-50_t.pkl \
                                OUTPUT_DIR ./exp_coco2x/supervised

python train_net.py --config-file ./configs/coco_R_50_FPN_2x_base.yaml \
                                --num-gpus 8 --dist-url tcp://127.0.0.1:1681 \
                                MODEL.WEIGHTS None \
                                OUTPUT_DIR ./exp_coco2x/supervised