for file in `find /media/jianghao/Samsung_T5/bilibili_test | grep ".*origvideo"`
do
    echo $(basename $file .origvideo)
    tmppath="/home/jianghao/Code/Graduation/4k1/videos/"
    echo $tmppath$(basename $file .origvideo)
    mkdir $tmppath$(basename $file .origvideo)
    /home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -i $file -pix_fmt yuv420p -frames 1000 -vf crop=3840:2144:0:0 decode.yuv -y # > /dev/null 2>&1

    /home/jianghao/anaconda3/envs/torch/bin/python /home/jianghao/Code/Graduation/4k1/codes/test_forward_yuv.py \
    -opt /home/jianghao/Code/Graduation/4k1/codes/options/test/test_EDSR_forward.yml #  > /dev/null 2>&1
	for qp in {22..37..5}
	do
		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -s 1920x1072 -pix_fmt yuv420p -i decode_net.yuv \
		-c:v libx265 -x265-params "preset=medium:psy=0:bframes=7:qp=${qp}" transcode_net.hevc -y > $tmppath$(basename $file .origvideo)/transcode_net_${qp} 2>&1

		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg  -s 3840x2144 -pix_fmt yuv420p -i decode.yuv -i transcode_net.hevc \
		-filter_complex "[1:v]scale=3840x2144:flags=bicubic[scale];[scale][0:v]psnr=stats_file="$tmppath$(basename $file .origvideo)/psnr_net_${qp} \
		-f null /dev/null > $tmppath$(basename $file .origvideo)/Avg_psnr_net_${qp} 2>&1
		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg  -s 3840x2144 -pix_fmt yuv420p -i decode.yuv -i transcode_net.hevc \
		-filter_complex "[1:v]scale=3840x2144:flags=bicubic[scale];[scale][0:v]ssim=stats_file="$tmppath$(basename $file .origvideo)/ssim_net_${qp} \
		-f null /dev/null > $tmppath$(basename $file .origvideo)/Avg_ssim_net_${qp} 2>&1

		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -s 3840x2144 -pix_fmt yuv420p -i decode.yuv -pix_fmt yuv420p -s 1920x1072 -sws_flags lanczos \
		-c:v libx265 -x265-params "preset=medium:psy=0:bframes=7:qp=${qp}" transcode_lanczos.hevc -y > $tmppath$(basename $file .origvideo)/transcode_lanczos_${qp} 2>&1

		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg  -s 3840x2144 -pix_fmt yuv420p -i decode.yuv -i transcode_lanczos.hevc \
		-filter_complex "[1:v]scale=3840x2144:flags=bicubic[scale];[scale][0:v]psnr=stats_file="$tmppath$(basename $file .origvideo)/psnr_lanczos_${qp} \
		-f null /dev/null > $tmppath$(basename $file .origvideo)/Avg_psnr_lanczos_${qp} 2>&1
		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg  -s 3840x2144 -pix_fmt yuv420p -i decode.yuv -i transcode_lanczos.hevc \
		-filter_complex "[1:v]scale=3840x2144:flags=bicubic[scale];[scale][0:v]ssim=stats_file="$tmppath$(basename $file .origvideo)/ssim_lanczos_${qp} \
		-f null /dev/null > $tmppath$(basename $file .origvideo)/Avg_ssim_lanczos_${qp} 2>&1
	done
	python BD-rate.py -path $tmppath$(basename $file .origvideo) >> BD-Rate-260000-22-37.log
done