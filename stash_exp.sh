dir_name=$(date +'Record-%m-%d-%H-%M')
# echo $dir_name
mkdir -p legacy/$dir_name
mv exp/* legacy/$dir_name
echo "Check ${dir_name}"