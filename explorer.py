import inspect
import re
import json

import gradio as gr
import timm
from timm.optim import create_optimizer_v2
from timm.scheduler.scheduler_factory import create_scheduler_v2
import plotly.express as px

import plotly.graph_objects as go



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


def get_current_lr(optimizer):
    """Get the current LR from the optimizer"""
    lrl = [param_group["lr"] for param_group in optimizer.param_groups]
    lr = sum(lrl) / len(lrl)
    return lr


def update_plot(name, lr, *args):
    default_args = get_scheduler_kwargs()
    override_args = dict(zip(default_args.keys(), args))

    # Convert string inputs back to their original types
    for arg_name, value in override_args.items():
        if isinstance(default_args[arg_name], (list, tuple)):
            override_args[arg_name] = [float(v.strip()) for v in value.split(",")]
        elif isinstance(default_args[arg_name], (int, float)):
            override_args[arg_name] = type(default_args[arg_name])(value)
        elif isinstance(default_args[arg_name], bool):
            override_args[arg_name] = bool(value)
        elif default_args[arg_name] is None and value.lower() == "none":
            override_args[arg_name] = None

    # Create dummy model and optimizer
    dummy_model = timm.create_model("resnet18")
    dummy_optimizer = create_optimizer_v2(dummy_model, opt="sgd", lr=lr)

    # Create scheduler
    scheduler, timm_num_epochs = create_scheduler_v2(
        optimizer=dummy_optimizer, sched=name, **override_args
    )

    # Simulate training loop
    epoch_ticks = []
    lrs = []
    for epoch in range(timm_num_epochs):
        scheduler.step(epoch=epoch)

        for batch_i in range(override_args["updates_per_epoch"]):
            global_batch = epoch * override_args["updates_per_epoch"] + batch_i
            global_epoch = global_batch / override_args["updates_per_epoch"]
            scheduler.step_update(num_updates=global_batch)
            epoch_ticks.append(global_epoch)
            lr_i = get_current_lr(dummy_optimizer)
            lrs.append(lr_i)

    # Create plot
    fig = px.scatter(x=epoch_ticks, y=lrs)
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(
        title=f"Learning rate schedule for {name}",
        xaxis_title="Epoch",
        yaxis_title="Learning rates",
    )

    # Pretty print kwargs
    kwargs_str = json.dumps(override_args, indent=2)
    kwargs_output = f"Scheduler: {name}\nLearning rate: {lr}\n\nKwargs:\n{kwargs_str}"

    return fig, kwargs_output


def create_interface():
    schedulers = get_timm_schedulers()
    default_args = get_scheduler_kwargs()

    with gr.Blocks(fill_width=True, title="timm LR scheduler explorer") as demo:
        gr.Markdown("# `timm` LR scheduler explorer")
        
        with gr.Row():
            with gr.Column(scale=1):
                name = gr.Dropdown(choices=schedulers, label="Scheduler", value=schedulers[0])
                lr = gr.Number(value=1.0, label="Learning rate")

                # Create input components for the first half of arguments
                override_args = {}
                args_list = list(default_args.items())
                midpoint = len(args_list) // 2

                for arg_name, default_value in args_list[:midpoint]:
                    if isinstance(default_value, bool):
                        override_args[arg_name] = gr.Checkbox(value=default_value, label=arg_name)
                    elif isinstance(default_value, (int, float)):
                        if arg_name == "updates_per_epoch":
                            default_value = 10
                        elif arg_name == "num_epochs":
                            default_value = 50
                        override_args[arg_name] = gr.Number(value=default_value, label=arg_name)
                    elif isinstance(default_value, str):
                        override_args[arg_name] = gr.Textbox(value=default_value, label=arg_name)
                    elif isinstance(default_value, (list, tuple)):
                        default_value = ",".join(str(v) for v in default_value)
                        override_args[arg_name] = gr.Textbox(value=default_value, label=arg_name)
                    elif default_value is None:
                        override_args[arg_name] = gr.Textbox(value="None", label=arg_name)
                    else:
                        raise NotImplementedError(f"Unknown type for {arg_name}")

            with gr.Column(scale=1):
                # Create input components for the second half of arguments
                for arg_name, default_value in args_list[midpoint:]:
                    if isinstance(default_value, bool):
                        override_args[arg_name] = gr.Checkbox(value=default_value, label=arg_name)
                    elif isinstance(default_value, (int, float)):
                        if arg_name == "updates_per_epoch":
                            default_value = 10
                        elif arg_name == "num_epochs":
                            default_value = 50
                        override_args[arg_name] = gr.Number(value=default_value, label=arg_name)
                    elif isinstance(default_value, str):
                        override_args[arg_name] = gr.Textbox(value=default_value, label=arg_name)
                    elif isinstance(default_value, (list, tuple)):
                        default_value = ",".join(str(v) for v in default_value)
                        override_args[arg_name] = gr.Textbox(value=default_value, label=arg_name)
                    elif default_value is None:
                        override_args[arg_name] = gr.Textbox(value="None", label=arg_name)
                    else:
                        raise NotImplementedError(f"Unknown type for {arg_name}")

            with gr.Column(scale=2):
                plot = gr.Plot()
                kwargs_output = gr.Textbox(label="Current Configuration", lines=10, interactive=False)

        # Set up event handlers for auto-update
        inputs = [name, lr] + list(override_args.values())
        for input_component in inputs:
            input_component.change(
                fn=update_plot,
                inputs=inputs,
                outputs=[plot, kwargs_output],
            )

        # Add on_load event to create initial plot and kwargs output
        demo.load(
            fn=update_plot,
            inputs=inputs,
            outputs=[plot, kwargs_output],
        )

        gr.Markdown("---")
        gr.Markdown("Fork me on [GitHub](https://github.com/crypdick/timm-lr-scheduler-explorer)")
        gr.Markdown("Made with ❤️ by [Richard Decal](richarddecal.com)")

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
