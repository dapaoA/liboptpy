for file in *.png
do
    newfile=$(echo $file | sed 's/png/eps/')
    echo $newfile
    convert $file eps2:$newfile
    mv $file png 
done
