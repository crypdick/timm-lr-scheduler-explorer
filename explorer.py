import inspect
import re

import streamlit as st
import timm
from timm.optim import create_optimizer_v2
from timm.scheduler.scheduler import Scheduler
from timm.scheduler.scheduler_factory import create_scheduler_v2
import matplotlib.pyplot as plt

# list all schedulers here
st.sidebar.markdown("# Learning rate scheduler")
@st.cache_data
def get_timm_schedulers():
    """Get list of timm schedulers

    Note: we cannot use the class names from timm.scheduler because they are not the same as the
    names used in timm.scheduler.scheduler_factory.create_scheduler_v2. Thus, we need to parse
    the create_scheduler_v2 function to get the list of schedulers.
    """
    code = inspect.getsource(create_scheduler_v2)
    schedulers = []
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("if sched ==") or line.startswith("elif sched =="):
            sched_name = line.split("==")[1]
            # extract the sceduler name using regex
            pattern = r"[a-zA-Z]+"
            sched_name = re.findall(pattern, sched_name)[0]
            schedulers.append( sched_name)
    assert len(schedulers)>3, "Error parsing scheduler names"
    return schedulers

name = st.sidebar.selectbox(
    "Scheduler",
    options=list(get_timm_schedulers()),
)
lr = st.sidebar.number_input("Learning rate", value=1.0)

@st.cache_data
def get_scheduler_kwargs():
    """Parse create_scheduler_v2 for args and their defaults"""
    # note: we cannot use the class names from timm.scheduler because they are not the same as the
    scheduler_args = inspect.signature(create_scheduler_v2).parameters
    scheduler_args = {
        arg_name: arg.default
        for arg_name, arg in scheduler_args.items()
        if arg_name not in {"optimizer", "sched"}
    }
    assert len(scheduler_args)>0, "Error parsing scheduler args"
    return scheduler_args

default_args = get_scheduler_kwargs()

# for each arg, create a widget that allows the user to override the default
override_args = {}
for arg_name, default_value in default_args.items():
    if isinstance(default_value, bool):
        value = st.sidebar.checkbox(arg_name, value=default_value)
    elif isinstance(default_value, (int, float)):
        value = st.sidebar.number_input(arg_name, value=default_value)
    elif isinstance(default_value, str):
        value = st.sidebar.text_input(arg_name, value=default_value)
    elif isinstance(default_value, (list, tuple)):
        # convert to string
        default_value = ",".join(str(v) for v in default_value)
        value = st.sidebar.text_input(arg_name, value=default_value)
        # convert back to original iterable and cast to original type
        original_type = type(default_value[0])
        if original_type in {int, float}:
            original_type = float  # avoid trouble when trying to convert default int value to a float
        value = [type(default_value)(original_type(v)) for v in value.split(",")]
    elif default_value is None:
        value = st.sidebar.text_input(arg_name, value=default_value)
        # convert back to None if text input is string "None"
        if value == "None":
            value = None
    else:
        raise NotImplementedError(f"Unknown type for {arg_name} {type(default_value)}: {default_value}")

    override_args[arg_name] = value

# create a scheduler with the user's args
dummy_model = timm.create_model("resnet18")
dummy_optimizer = create_optimizer_v2(dummy_model, opt="sgd", lr=lr)


scheduler, _ = create_scheduler_v2(optimizer=dummy_optimizer, sched=name, **override_args)
assert isinstance(scheduler, Scheduler), f"Expected scheduler to be a timm.scheduler.Scheduler, got {type(scheduler)}"

# simulate a training loop with the scheduler using num_epochs value and plot the LR
num_epochs = override_args["num_epochs"]
lrs = []
lrs2 = []
if override_args["cycle_mul"] > 1:
    num_epochs = int(num_epochs * override_args["cycle_mul"])
for epoch in range(num_epochs):
    scheduler.step(epoch)
    # get the LR for this epoch from the optimizer
    lr = dummy_optimizer.param_groups[0]["lr"]
    #assert isinstance(lr, float), f"Expected LR to be a float, got {type(lr)}"
    lrs.append(lr)
    lrs2.append(scheduler._get_lr(epoch))
    # lrs2.append(scheduler.get_epoch_values(epoch))
assert len(lrs) == num_epochs, f"Expected {num_epochs} LR values, got {len(lrs)}"

fig = plt.figure()
plt.plot(lrs)
plt.title(f"Learning rate schedule for {name}")
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
st.pyplot(fig)


fig = plt.figure()
plt.plot(lrs2)
plt.title(f"Learning rate schedule for {name}")
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
st.pyplot(fig)