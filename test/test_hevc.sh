#for name in People Traffic;
#for name in FourPeople Kristen Johnny;
#for name in BasketballPass BlowingBubbles BQSquare RaceHorses;
for name in  BasketballDrill PartyScene RaceHorses BQMall;
do
    GT_path="/home/gy4qf62/HEVCTestSequences/ClassC/${name}"
    INV_root="/home/gy4qf62/Invertible-Image-Rescaling/results/${name}"
    logdir=log/ClassC_str4/${name}
    
    sed -i "1c name: ${name}" codes/options/test/test_EDSR_forward.yml
    sed -i "13s/HEVCTestSequences\/.*/HEVCTestSequences\/ClassC\/${name}/g" codes/options/test/test_EDSR_forward.yml
    
    sed -i "1c name: ${name}" codes/options/test/test_EDSR_backward.yml
    sed -i "13s/results\/.*\//results\/${name}\//g" codes/options/test/test_EDSR_backward.yml
    
    sed -i "1c name: ${name}" codes/options/test/test_EDSR_backward2.yml
    
    sed -i "1c name: ${name}" codes/options/test/test_bicubic_down.yml
    sed -i "13s/HEVCTestSequences\/.*/HEVCTestSequences\/ClassC\/${name}/g" codes/options/test/test_bicubic_down.yml
    
    sed -i "1c name: ${name}" codes/options/test/test_bicubic_up.yml
    sed -i "13s/results\/.*\//results\/${name}\//g" codes/options/test/test_bicubic_up.yml
    
    mkdir -p ${logdir}
    
    mkdir -p ${INV_root}/decode_lr
    mkdir -p ${INV_root}/decode_lr_bic
    mkdir -p ${INV_root}/decode_GT
    
    #ffmpeg -s 1920x1080 -pix_fmt yuv420p -i  ${GT_path}/${name}.yuv -vf crop=1920:1072 ${GT_path}/${name}_2.yuv -y
    ffmpeg -s 832x480 -pix_fmt yuv420p -i  ${GT_path}/${name}.yuv "${GT_path}/im%04d.png"
    python codes/test_forward.py -opt codes/options/test/test_EDSR_forward.yml
    #python codes/test_bic.py -opt codes/options/test/test_bicubic_down.yml
    ffmpeg -i "${INV_root}/lr/im%04d.png" -pix_fmt yuv420p /home/gy4qf62/HM-16.20/test/${name}.yuv -y
    for QP in 32 37 42 47;
    do
        /home/gy4qf62/HM-16.20/bin/TAppEncoderStatic -c /home/gy4qf62/HM-16.20/test/ClassC/${name}.cfg -i /home/gy4qf62/HM-16.20/test/${name}.yuv -c /home/gy4qf62/HM-16.20/test/encoder_intra_main.cfg -b ${name}.hevc -hgt 240 -wdt 416 > ${logdir}/${QP}_HMlog_AI.txt -q ${QP} 2>&1
        ffmpeg -i /home/gy4qf62/Invertible-Image-Rescaling/${name}.hevc "${INV_root}/decode_lr_bic/im%04d.png"
        sed -i "1c name: ${name}" codes/options/test/test_EDSR_backward.yml
        sed -i "13s/results\/.*\//results\/${name}\//g" codes/options/test/test_EDSR_backward.yml
        python codes/test_backward.py -opt codes/options/test/test_EDSR_backward.yml
        python metrics/calculate_PSNR_SSIM.py -folder_GT ${GT_path} -folder_Gen "${INV_root}/sr" -suffix "" > ${logdir}/${QP}_psnr_AI.txt 2>&1
        
        
        /home/gy4qf62/HM-16.20/bin/TAppEncoderStatic -c /home/gy4qf62/HM-16.20/test/ClassC/${name}.cfg -i /home/gy4qf62/HM-16.20/test/${name}.yuv -c /home/gy4qf62/HM-16.20/test/encoder_lowdelay_P_main.cfg -b ${name}.hevc -hgt 240 -wdt 416 > ${logdir}/${QP}_HMlog_LDP.txt -q ${QP} 2>&1
        ffmpeg -i /home/gy4qf62/Invertible-Image-Rescaling/${name}.hevc "${INV_root}/decode_lr_bic/im%04d.png"
        sed -i "1c name: ${name}" codes/options/test/test_EDSR_backward.yml
        sed -i "13s/results\/.*\//results\/${name}\//g" codes/options/test/test_EDSR_backward.yml
        python codes/test_backward.py -opt codes/options/test/test_EDSR_backward.yml
        python metrics/calculate_PSNR_SSIM.py -folder_GT ${GT_path} -folder_Gen "${INV_root}/sr" -suffix "" > ${logdir}/${QP}_psnr_LDP.txt 2>&1
        
        
        /home/gy4qf62/HM-16.20/bin/TAppEncoderStatic -c /home/gy4qf62/HM-16.20/test/ClassC/${name}.cfg -i /home/gy4qf62/HM-16.20/test/${name}.yuv -c /home/gy4qf62/HM-16.20/test/encoder_randomaccess_main.cfg -b ${name}.hevc -hgt 240 -wdt 416 > ${logdir}/${QP}_HMlog_RA.txt -q ${QP} 2>&1
        ffmpeg -i /home/gy4qf62/Invertible-Image-Rescaling/${name}.hevc "${INV_root}/decode_lr_bic/im%04d.png"
        sed -i "1c name: ${name}" codes/options/test/test_EDSR_backward.yml
        sed -i "13s/results\/.*\//results\/${name}\//g" codes/options/test/test_EDSR_backward.yml
        python codes/test_backward.py -opt codes/options/test/test_EDSR_backward.yml
        python metrics/calculate_PSNR_SSIM.py -folder_GT ${GT_path} -folder_Gen "${INV_root}/sr" -suffix "" > ${logdir}/${QP}_psnr_RA.txt 2>&1
    done
    
done
