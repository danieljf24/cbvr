for collection in track_1_shows  track_2_movies
do 
    for feat in inception-pool3 c3d-pool5
    do
    python feature_convert.py $collection $feat
    done
done