"""
Memory Debugging Utilities for DINOv2 Object Detection
"""
import torch
import gc

def print_tensors_by_size():
    """Print all tensors sorted by size to find memory leaks"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
        
    # Get all objects tracked by the garbage collector
    objs = gc.get_objects()
    tensors = []
    
    # Filter for tensors
    for obj in objs:
        try:
            if torch.is_tensor(obj) and obj.device.type == 'cuda':
                tensors.append(obj)
        except:
            pass  # Skip problematic objects
    
    if not tensors:
        print("No CUDA tensors found")
        return
    
    # Sort by size (bytes)
    tensors.sort(key=lambda x: x.nelement() * x.element_size(), reverse=True)
    
    # Print top 20 tensors
    print("\n==== LARGEST TENSORS IN MEMORY ====")
    total_bytes = 0
    for i, tensor in enumerate(tensors[:20]):
        size_mb = tensor.nelement() * tensor.element_size() / (1024 * 1024)
        total_bytes += tensor.nelement() * tensor.element_size()
        shape_str = 'x'.join(str(dim) for dim in tensor.shape)
        print(f"{i+1}. Size: {size_mb:.2f} MB, Shape: {shape_str}, Type: {tensor.dtype}, Requires grad: {tensor.requires_grad}")
    
    # Calculate total CUDA memory
    total_memory = sum(t.nelement() * t.element_size() for t in tensors)
    print(f"Total tensor memory: {total_memory / (1024 * 1024):.2f} MB")
    print(f"Total CUDA tensors: {len(tensors)}")
    print("====================================\n")

def clear_memory(model=None):
    """Clear memory aggressively"""
    # Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run garbage collector
    gc.collect()
    
    # Delete unused variables
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == 'cuda':
                if model is not None:
                    # Check if tensor belongs to the model
                    is_model_param = False
                    for param in model.parameters():
                        if obj.data_ptr() == param.data_ptr():
                            is_model_param = True
                            break
                    if not is_model_param:
                        del obj
                else:
                    del obj
        except:
            pass

def memory_stats(device):
    """Return formatted memory stats for the specified device"""
    if not torch.cuda.is_available() or device.type != 'cuda':
        return "CUDA not available"
    
    allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
    
    return (f"CUDA Memory: {allocated:.2f}MB (allocated), {max_allocated:.2f}MB (peak allocated), "
            f"{reserved:.2f}MB (reserved)")