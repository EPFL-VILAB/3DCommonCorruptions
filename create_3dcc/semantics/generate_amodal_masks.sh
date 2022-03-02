echo "creating occluded masks"
blender -b --enable-autoexec -noaudio --python ./create_semantic_images_replica.py -- MODEL_PATH=$PWD/../../sample_data/semantics MODEL_FILE=mesh_semantic.ply $1 1 $2

echo "creating unoccluded masks"
blender -b --enable-autoexec -noaudio --python ./create_semantic_images_replica.py -- MODEL_PATH=$PWD/../../sample_data/semantics MODEL_FILE=mesh_semantic.ply $1 0 $2

echo "overlaying masks on rgb images"
python create_amodal_masks.py --object_class $2 --set_id $1
