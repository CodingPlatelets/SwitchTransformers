import torch
import time
from switch_transformers.model import SwitchTransformer
from torch.profiler import profile, record_function, ProfilerActivity

start_all = time.time()
# Generate a random tensor of shape (1, 10) with values between 0 and 100
start = time.time()
x = torch.randint(0, 1_000_000, (8, 128)).cuda()
end = time.time()
print(f"the modle generate time is {end - start}")
print()

# Create an instance of the SwitchTransformer model
# num_tokens: the number of tokens in the input sequence
# dim: the dimensionality of the model
# heads: the number of attention heads
# dim_head: the dimensionality of each attention head
model = (
    SwitchTransformer(
        num_tokens=1_000_000,
        dim=1024,
        heads=8,
        dim_head=128,
        depth=10,
    )
    .cuda()
    .eval()
)


# Pass the input tensor through the model
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
) as prof:
    out = model(x)
end_all = time.time()

print(f"the total time is {end_all - start_all}")
# Print the shape of the output tensor
print(prof.key_averages().table(sort_by="cuda_time_total"))

# print(prof.key_averages().table(sort_by="cuda_memory_usage"))

prof.export_chrome_trace("trace.json")
print(out.shape)
