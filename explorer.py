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
    """
    Get list of timm schedulers

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
            schedulers.append(sched_name)
    assert len(schedulers) > 3, "Error parsing scheduler names"
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
    assert len(scheduler_args) > 0, "Error parsing scheduler args"
    return scheduler_args


default_args = get_scheduler_kwargs()

# for each arg, create a widget that allows the user to override the default
override_args = {}
for arg_name, default_value in default_args.items():
    if isinstance(default_value, bool):
        value = st.sidebar.checkbox(arg_name, value=default_value)
    elif isinstance(default_value, (int, float)):
        # exception: set default of updates_per_epoch to 100 instead of the default 0
        if arg_name == "updates_per_epoch":
            st.sidebar.markdown(
                f"Note: timm default for {arg_name}={default_value}; overriding to 10 here"
            )
            default_value = 10
        # exception: set default of num_epochs to 100 instead of the default 300
        if arg_name == "num_epochs":
            st.sidebar.markdown(
                f"Note: timm default for {arg_name}={default_value}; overriding to 50 here"
            )
            default_value = 50
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
        raise NotImplementedError(
            f"Unknown type for {arg_name} {type(default_value)}: {default_value}"
        )

    override_args[arg_name] = value


# create a scheduler with the user's args
dummy_model = timm.create_model("resnet18")
dummy_optimizer = create_optimizer_v2(dummy_model, opt="sgd", lr=lr)


scheduler, timm_num_epochs = create_scheduler_v2(
    optimizer=dummy_optimizer, sched=name, **override_args
)


# simulate a training loop with the scheduler using num_epochs value and plot the LR


def get_current_lr(optimizer):
    """Get the current LR from the optimizer"""
    lrl = [param_group["lr"] for param_group in optimizer.param_groups]
    lr = sum(lrl) / len(lrl)
    return lr


epoch_ticks = []
lrs = []
lrs2 = []
for epoch in range(timm_num_epochs):
    # note: need to step at the beginning of epoch, since we use 0-indexing for epoch
    scheduler.step(epoch=epoch)  # will have no effect if step_on_epochs is False

    epoch_ticks.append(epoch)
    lr_i = get_current_lr(dummy_optimizer)
    lrs.append(lr_i)

    if override_args[
        "step_on_epochs"
    ]:  # sanity check: get_lr should be the same as optimizer lr
        assert (
            lr_i == scheduler._get_lr(epoch)[0]
        ), f"Mismatch between optimizer LR {lr_i} and scheduler LR {scheduler._get_lr(epoch)}"

    # use 1-indexing for batch_i to get the expected num batches
    for batch_i in range(1, override_args["updates_per_epoch"]):
        global_batch = epoch * override_args["updates_per_epoch"] + batch_i
        global_epoch = global_batch / override_args["updates_per_epoch"]
        # will have no effect if step_on_epochs is True
        scheduler.step_update(num_updates=global_batch)
        epoch_ticks.append(global_epoch)
        lr_i = get_current_lr(dummy_optimizer)
        lrs.append(lr_i)

        if not override_args[
            "step_on_epochs"
        ]:  # sanity check: get_lr should be the same as optimizer lr
            assert (
                lr_i == scheduler._get_lr(global_batch)[0]
            ), f"Mismatch between optimizer LR {lr_i} and scheduler LR {scheduler._get_lr(global_batch)}"


assert len(lrs) == timm_num_epochs * override_args["updates_per_epoch"], (
    f"Expected {timm_num_epochs * override_args['updates_per_epoch']-1} LR values, "
    f"got {len(lrs)}"
)

fig = plt.figure()
plt.scatter(epoch_ticks, lrs, color="red", s=1, marker=".")
plt.title(f"Learning rate schedule for {name}")
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.ylim(bottom=0)
st.pyplot(fig)

st.markdown(f"Args used for {name}: {override_args}")
