#!/bin/bash

#   This shell script must run in /data directory on develop env, if not, you can read this code.
#   It compresses model_serving and slu-serving, then scp the compressed file to the host 10.153.169.115.
#   when login 115, tar and backup the model and serving file, then decompress new file, restart two serving.


usage(){
   cat <<HELP_USAGE
usage: $0 -m <msg>
-m <msg>, --message=<msg>  Use the given <msg> as the commit message
                           of new model or serving
-h ,--help
HELP_USAGE
}

TEMP=`getopt -l "message:,help" -o "m:h" -a -- "$@"`

if [ $? != 0 ] ; then echo "Terminating..." >&2 ; usage; exit 1 ; fi

# Note the quotes around `$TEMP': they are essential!
#set 会重新排列参数的顺序，也就是改变$1,$2...$n的值，这些值在getopt中重新排列过了
eval set -- "$TEMP"
#echo "$TEMP"
#echo $0 $1 $2 $3

#  不太明白没有参数传入时为什么是--. 后发现参数都是以--结尾
#  当没有参数传入时,即只有--时，报错提醒，&&关系预算符代替if，前面正确后面运算
[[ $1 == -- ]] && {
    echo -e "error: need at least one argument\n"
    usage;exit 1
}

#经过getopt的处理，下面处理具体选项。

while true ; do
    case "$1" in
        -m|--message)
            case "$2" in
                "")
                    echo -e "switch 'm' requires a value \n"
                    usage; exit 1 ;;
                *)  msg="$2"; shift 2 ;;
            esac ;;
        -h|--help)
            usage
            exit 0 ;;
        --) shift ; break ;;
        *) echo "Internal error!" ; exit 1 ;;
    esac
done

DATE=$(date +%m_%d)
filename="modelWithServing_${DATE}_$msg.tar.gz"

echo -e "************start compress model_serving and slu-serving:************\n"
tar cvzf update_history/$filename model_serving slu-serving
echo -e "done\n"

echo -e "**************start scp file to 10.153.169.115:/data***************\n"
scp update_history/$filename root@10.153.169.115:/data/update_history
echo -e "done\n"

echo -e "**************start backup, decompress file, change ip ,restart server*************\n"
#远程执行命令包含$ 时要进行转义才能传过去，不然会被解释后传过去如$filename
ssh root@10.153.169.115  2>&1 <<remote_run
cd /data
tar -xvzf ./update_history/$filename
cd slu-serving
sed -i 's/10.16.168.82/10.153.169.115/g' app/configuration.py
ps aux|grep tensorflow|grep -v grep|awk '{print \$2}'|xargs kill -9
ps aux|grep "python[3]* main"|grep -v grep|awk '{print \$2}'|xargs kill -9
bash restart_serve.sh
remote_run
echo -e "***********************done***********************\n"
echo "!!!NOTICE!!! NOW YOU SHOULD check all processes whether work normally
      and use Postman to test some cases, then write update log"


#screen -dmS serving
#screen -list
#screen -r serving
#nohup tensorflow_model_server --port=9001 --model_config_file=/data/model_serving/model_serving.conf &
#screen tensorflow_model_server --port=9001 --model_config_file=/data/model_serving/model_serving.conf
#nohup python3 main.py &
#screen python3 main.py