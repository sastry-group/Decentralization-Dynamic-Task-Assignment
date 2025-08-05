import h5py

def explore_jld2_file(file_path):
    with h5py.File(file_path, 'r') as f:
        def recurse(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"[Dataset] {name}: shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"[Group] {name}")

        print("Contents of the JLD2 file:")
        f.visititems(recurse)

# Example usage
# jld2_file = "sf_halton_tt_estimates_scoba.jld2"
jld2_file = "./param_files/sf_halton_tt_estimates_scoba.jld2"
explore_jld2_file(jld2_file)