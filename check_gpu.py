# check_gpu.py
import dlib

try:
    # This constant is True if dlib was compiled with CUDA support
    is_cuda_supported = dlib.DLIB_USE_CUDA
    print(f"Is dlib compiled with CUDA support? {is_cuda_supported}")

    if is_cuda_supported:
        # This function will list the number of GPUs dlib can see.
        # It will throw an error if the CUDA drivers can't be found or initialized.
        device_count = dlib.cuda.get_num_devices()
        print(f"Number of CUDA devices found: {device_count}")

        if device_count > 0:
            print("\nSuccess! Your dlib installation can see the GPU.")
            print("The 'cnn' model should now be running on your NVIDIA GPU.")
        else:
            print("\nWarning: dlib is compiled for CUDA, but no active GPU was found.")
            print("This might be an issue with your NVIDIA driver or CUDA Toolkit installation.")

    else:
        print("\nError: Your dlib installation was not compiled with CUDA support.")
        print("This means the compiler did not find your CUDA installation when you installed dlib.")

except Exception as e:
    print(f"\nAn error occurred while checking for CUDA support: {e}")
    print("This usually points to a problem with the NVIDIA driver or a missing component like cuDNN.") 