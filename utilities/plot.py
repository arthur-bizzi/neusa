import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from matplotlib import cm
import matplotlib.ticker as mtick

def create_graphs_grid(x_gt, u_gt, u_pred, titles, figs_dir):
    _, axs = plt.subplots(2,2, figsize=(10,8))

    for i, ax in enumerate(axs.flat):
        ax.plot(x_gt,u_gt[i],label="GT", color='blue')
        ax.plot(x_gt,u_pred[i],label="Predicted", color='red')
        ax.set_title(titles[i])
        ax.legend()
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir,"t_grid.png"))

def create_time_evolution_gif(x_gt,t_gt,u_gt,u_pred,figs_dir, gif_name="evolution.gif", duration=50):
    
    ymin, ymax = np.min(u_gt), np.max(u_gt)
    
    frames = []
    for i in range(len(t_gt)):
        fig, ax = plt.subplots()
        ax.plot(x_gt,u_gt[i,:],color='blue', label="GT")
        ax.plot(x_gt,u_pred[i,:],color='red', label="Predicted")
        ax.set_title(f"t={t_gt[i].item()}")
        ax.set_ylim(ymin,ymax+0.05)
        ax.grid(False)
    
        fname = f"_frame_{i:03d}.png"
        fig.savefig(fname)
        plt.close(fig)
        frames.append(Image.open(fname))
    
    frames[0].save(os.path.join(figs_dir,gif_name),save_all=True,append_images=frames[1:],duration=duration,loop=0)

    for frame in frames:
        frame.close()
    for i in range(len(t_gt)):
        os.remove(f"_frame_{i:03d}.png")


def create_time_evolution_2d_gif(ts,u,figs_dir, gif_name="evolution.gif",extent=[0,1,0,1], duration=50):
    
    
    frames = []
    for i in range(len(ts)):
        fig, axs = plt.subplots(figsize=(6,5))
        im0 = axs.imshow(u[i,:,:].T,origin='lower', cmap='seismic', extent=extent)
        axs.set_title(f"Prediction at t={ts[i]:.2f}")
        axs.axis('off')

        fname = f"_frame_{i:03d}.png"
        fig.savefig(fname,dpi=300)
        plt.close(fig)
        frames.append(Image.open(fname))
    
    frames[0].save(os.path.join(figs_dir,gif_name),save_all=True,append_images=frames[1:],duration=duration,loop=0)

    for frame in frames:
        frame.close()
    for i in range(len(ts)):
        os.remove(f"_frame_{i:03d}.png")

def plot_prediction_vs_groundtruth(ts,u_pred,u_gt,extent=[0, 1, 0, 1],save_path=None,cmap="seismic"):
    
    num_frames = 5
    idxs = np.linspace(0, len(ts)-1, num_frames, dtype=int)
    selected_ts = [ts[i] for i in idxs]

    fig, axs = plt.subplots(2, num_frames, figsize=(num_frames * 3, 6))
    
    vmax = np.abs(u_gt).max()
    vmin = -vmax
    
    for j, i in enumerate(idxs):
        im1 = axs[0, j].imshow(u_pred[i].T, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
        axs[0, j].set_title(f"t = {selected_ts[j]:.2f}")
        axs[0, j].set_xticks([])
        axs[0, j].set_yticks([])
        axs[0, j].tick_params(labelbottom=False, labelleft=False)

        im2 = axs[1, j].imshow(u_gt[i].T, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
        axs[1, j].set_xticks([])
        axs[1, j].set_yticks([])
        axs[1, j].tick_params(labelbottom=False, labelleft=False)

    axs[0, 0].set_ylabel("Prediction", fontsize=12)
    axs[1, 0].set_ylabel("Ground-Truth", fontsize=12)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # left, bottom, width, height
    fig.colorbar(im1, cax=cbar_ax).set_label("Amplitude")

    plt.subplots_adjust(right=0.9, wspace=0.1, hspace=0.2)


    if save_path:
        plt.savefig(save_path, dpi=300)

def plot_error(ts, error, extent=[0, 1, 0, 1], save_path=None, cmap="seismic"):
    num_frames = 5
    idxs = np.linspace(0, len(ts)-1, num_frames, dtype=int)
    selected_ts = [ts[i] for i in idxs]

    fig, axs = plt.subplots(1, num_frames, figsize=(num_frames * 3, 6))

    # garantir que axs seja sempre um array
    if num_frames == 1:
        axs = [axs]

    vmin = error.min()
    vmax = error.max()

    for j, i in enumerate(idxs):
        im = axs[j].imshow(error[i].T, cmap=cmap,
                           extent=extent, vmin=vmin, vmax=vmax)
        axs[j].set_title(f"t = {selected_ts[j]:.2f}")
        axs[j].set_xticks([])
        axs[j].set_yticks([])
        axs[j].tick_params(labelbottom=False, labelleft=False)

    axs[0].set_ylabel("Error", fontsize=12)

    # barra de cores
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
    fig.colorbar(im, cax=cbar_ax).set_label("Amplitude")

    plt.subplots_adjust(right=0.9, wspace=0.1)

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()



def plot_instant(x,y,u, projection="rectilinear", cmap=cm.seismic, title="",angles=(60,240,None), y_name="y",x_name="x", axes=None):
    
    if axes is None:
        fig,axes=plt.subplots(subplot_kw={"projection":projection})
        axes.set_title(title)
    else:
        fig=axes.get_figure()
        assert axes.name==projection
    fig.set_dpi(1200)
    fig.set_size_inches(5.5,4)
    axes.set_xlabel(x_name)
    axes.set_ylabel(y_name)
    # max_val=np.max(np.abs(u))
    max_val=np.quantile(np.abs(u),0.99)
    vmin,vmax=-max_val, +max_val
    if projection=="rectilinear":
        u_plot = axes.pcolormesh(x,y,u.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
    elif projection=="3d":
        x_mesh,y_mesh = np.meshgrid(x,y,indexing="ij")
        u_plot = axes.plot_surface(x_mesh,y_mesh,u, cmap=cmap, vmin=vmin, vmax=vmax)
        axes.view_init(elev=angles[0], azim=angles[1], roll=angles[2])
        axes.zaxis.set_major_formatter(mtick.NullFormatter())
    cbar=fig.colorbar(u_plot, shrink=1.0, ax=axes)

    return fig

def figures_2d_subplots(t,x,y,u_pred,u_gt, folder, snapshot_times=(), order="txy", model_name="PINN", projection="rectilinear", x_name="x", y_name="y"):
    if not os.path.exists(folder): os.makedirs(folder)

    cmap=cm.seismic
    indices_t = []
    times = []
    for i, snapshot_time in enumerate(snapshot_times):
        ind_t = len(np.nonzero(snapshot_time>=t)[0])-1
        indices_t.append(ind_t)
        times.append(t[ind_t])
    
    ##Checking the shapes of the provided vectors
    if order=="txy":
        assert np.all(u_gt.shape[:-1]==np.concatenate((t.shape,x.shape,y.shape)))
        assert u_gt.shape==u_pred.shape
        u_dimensions = [(u_gt[indices_t,:,:,i], u_pred[indices_t,:,:,i]) for i in range(u_gt.shape[-1])] ##If the output is 2d, divides it in components U1 and U2
    elif order=="xyt":
        assert np.all(u_gt.shape[:-1]==np.concatenate((x.shape,y.shape, t.shape)))
        assert u_gt.shape==u_pred.shape
        u_dimensions = [(u_gt[:,:,indices_t,i], u_pred[:,:,indices_t,i]) for i in range(u_gt.shape[-1])] ##If the output is 2d, divides it in components U1 and U2
    if len(u_dimensions)==1: u_dimension_names = ["U"]
    elif len(u_dimensions)<=3: u_dimension_names= ["U","V","W"]
    else: u_dimension_names = [f"U{i}" for i in range(len(u_dimensions))]

    for (i,(gt,pred)) in enumerate(u_dimensions):
        u_name = u_dimension_names[i]

        ##Solution plot
        fig, axes = plt.subplots(len(snapshot_times),2, subplot_kw={"projection":projection})
        fig.set_dpi(1200)
        fig.subplots_adjust(hspace=0.4,wspace=0.15)
        fig.set_size_inches(1+2*len(u_dimensions),5*len(snapshot_times))
        for (i, time) in enumerate(times):
            plot_instant(x,y,gt[i], projection=projection, cmap=cmap, title=f"Ground Truth {u_name}", axes=axes[i,0], x_name=x_name, y_name=y_name)
            plot_instant(x,y,pred[i], projection=projection, cmap=cmap, title=f"{model_name} {u_name}", axes=axes[i,1], x_name=x_name, y_name=y_name)
            axes[i,0].set_ylabel(y_name)
            axes[i,1].set_ylabel("")
            axes[i,1].yaxis.set_major_formatter(mtick.NullFormatter())
            for j in (0,1):
                axes[i,j].set_xlabel("")
                axes[i,j].xaxis.set_major_formatter(mtick.NullFormatter())  

            axes[i,0].set_title(f"Ground Truth {u_name}: t={time:.2f}")
            axes[i,1].set_title(f"{model_name} {u_name}: t={time:.2f}")
        for j in (0,1):
            axes[-1,j].set_xlabel(x_name)
        fig.savefig(os.path.join(folder,f"{model_name}_{u_name}.png"), transparent=True, format="png", bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(len(snapshot_times),1, subplot_kw={"projection":projection})
        fig.set_dpi(1200)
        fig.subplots_adjust(hspace=0.3,wspace=0.15)
        fig.set_size_inches(3,5*len(snapshot_times))
        for (i, time) in enumerate(times):
            error = pred[i]-gt[i]
            plot_instant(x,y,error, projection=projection, cmap=cmap, title=f"{model_name} Error {u_name}", axes=axes[i], x_name=x_name, y_name=y_name)
            # plot_instant(x,y,gt[i], projection=projection, cmap=cmap, title=f"Ground Truth {u_name}", axes=axes[i,0])
            # plot_instant(x,y,pred[i], projection=projection, cmap=cmap, title=f"{model_name} {u_name}", axes=axes[i,1])
            axes[i].set_ylabel("y")
            axes[i].set_xlabel("")
            axes[i].xaxis.set_major_formatter(mtick.NullFormatter())  

            axes[i].set_title(f"Error {u_name}: t={time:.2f}")
        axes[-1].set_xlabel("x")
        fig.savefig(os.path.join(folder,f"{model_name}_{u_name}_error.png"), transparent=True, format="png", bbox_inches="tight")
        plt.close(fig)


def generate_figures_1d(t,x,u_pred,u_gt,folder,order="tx", model_name="PINN", x_name="x"):
    '''
    Generates figures for representing the solution as a heatmap across the full time series
    '''
    if not os.path.exists(folder): os.makedirs(folder)

    cmap=cm.seismic
    ##Checking the shapes of the provided vectors
    if order=="tx":
        assert np.all(u_gt.shape[:-1]==np.concatenate((t.shape,x.shape)))
        assert u_gt.shape==u_pred.shape
        u_dimensions = [(u_gt[...,i].T, u_pred[...,i].T) for i in range(u_gt.shape[-1])] ##If the output is 2d, divides it in components U1 and U2
    elif order=="xt":
        assert np.all(u_gt.shape[:-1]==np.concatenate((x.shape, t.shape)))
        assert u_gt.shape==u_pred.shape
        u_dimensions = [(u_gt[...,i], u_pred[...,i]) for i in range(u_gt.shape[-1])] ##If the output is 2d, divides it in components U1 and U2
    if len(u_dimensions)==1: u_dimension_names = ["U"]
    elif len(u_dimensions)<=3: u_dimension_names= ["U","V","W"]
    else: u_dimension_names = [f"U{i}" for i in range(len(u_dimensions))]
    
    for (i,(gt,pred)) in enumerate(u_dimensions):
        ##Plot 2D plot of solution and GT
        u_name = u_dimension_names[i]
        fig_gt = plot_instant(x,t,gt, projection="rectilinear", cmap=cmap, title=f"Ground Truth {u_name}", y_name="t", x_name=x_name)
        fig_gt.savefig(os.path.join(folder,f"GT_{u_name}.png"), transparent=True, format="png", bbox_inches="tight")
        fig_pred = plot_instant(x,t,pred, projection="rectilinear", cmap=cmap, title=f"{model_name} {u_name}", y_name="t", x_name=x_name)
        fig_pred.savefig(os.path.join(folder,f"{model_name}_{u_name}_pred.png"), transparent=True, format="png", bbox_inches="tight")
        plt.close(fig_gt)
        plt.close(fig_pred)
        ##Plot 2D plot of error
        error = pred-gt
        fig_error = plot_instant(x,t,error, projection="rectilinear", cmap=cmap, title=f"{model_name} Error", y_name="t", x_name=x_name)
        fig_error.savefig(os.path.join(folder,f"{model_name}_{u_name}_error.png"), transparent=True, format="png", bbox_inches="tight")
        plt.close(fig_error)

def generate_figures_3d(t, x, y, z, uv_pred, uv_gt, save_dir_figs, order="txyz", model_name="NeuSA"):
    # TODO: implement a nice plot for 3D data
    if order == "xyzt":
        raise NotImplementedError("3D plotting for order 'xyzt' is not implemented yet.")
    else:
        assert order == "txyz", "Only 'txyz' index order is supported for 3D plotting."
    
    idx_zero = np.where(x == 0)[0]
    if len(idx_zero) == 0:
        raise ValueError("x does not contain 0.")
    position_x = idx_zero[0]

    idy_zero = np.where(y == 0)[0]
    if len(idy_zero) == 0:
        raise ValueError("y does not contain 0.")
    position_y = idy_zero[0]

    idz_zero = np.where(z == 0)[0]
    if len(idz_zero) == 0:
        raise ValueError("z does not contain 0.")
    position_z = idz_zero[0]

    # plot in (x,t)
    xt_path = os.path.join(save_dir_figs, "xt plane")
    uv_pred_x = uv_pred[:,:,position_y,position_z]
    uv_gt_x = uv_gt[:,:,position_y,position_z]
    generate_figures_1d(t,x,uv_pred_x,uv_gt_x, xt_path, order="tx", model_name=model_name, x_name="x")

    # plot in (y,t)
    yt_path = os.path.join(save_dir_figs, "yt plane")
    uv_pred_y = uv_pred[:,position_x,:,position_z]
    uv_gt_y = uv_gt[:,position_x,:,position_z]
    generate_figures_1d(t,y,uv_pred_y,uv_gt_y, yt_path, order="tx", model_name=model_name, x_name="y")

    # plot in (z,t)
    zt_path = os.path.join(save_dir_figs, "zt plane")
    uv_pred_z = uv_pred[:,position_x,position_y,:]
    uv_gt_z = uv_gt[:,position_x,position_y,:]
    generate_figures_1d(t,z,uv_pred_z,uv_gt_z, zt_path, order="tx", model_name=model_name, x_name="z")

    # Plot 2D sections for a few key instants
    snapshot_times = [t[0], t[len(t)//2], t[-1]]  # Example: first, middle, and last time instants
    sections_path = os.path.join(save_dir_figs, "xy_sections")
    figures_2d_subplots(t, x, y, uv_pred[:,:,:,position_z], uv_gt[:,:,:,position_z], sections_path, snapshot_times=snapshot_times, order="txy", model_name=model_name, projection="rectilinear")
    # Plot 2D sections for xz plane at a few key instants
    xz_path = os.path.join(save_dir_figs, "xz_sections")
    uv_pred_xz = uv_pred[:, :, position_y, :]
    uv_gt_xz = uv_gt[:, :, position_y, :]
    figures_2d_subplots(t, x, z, uv_pred_xz, uv_gt_xz, xz_path, snapshot_times=snapshot_times, order="txy", model_name=model_name, projection="rectilinear", x_name="x", y_name="z")



def plot_extrapolation_rl2(ts,rl2s,path):
    plt.figure(figsize=(10,10))
    plt.plot(ts,rl2s)
    # plt.yscale('log')
    plt.xlabel("time")
    plt.ylabel("rl2")
    plt.savefig(path)

def plot_extrapolation_rl1(ts,rl1s,path):
    plt.figure(figsize=(10,10))
    plt.plot(ts,rl1s)
    plt.yscale('log')
    plt.xlabel("time")
    plt.ylabel("rl1")
    plt.savefig(path)


def plot_comparison(extrapolated_v, ground_truth_v, relative_l2_errors, time, t_target=2.0, model_order = ("QRes", "NeuSA", "PINN"),
                    cmap="coolwarm"):
    t_idx = np.argmin(np.abs(time - t_target))  # Find index closest to t=t_target

    x = np.linspace(0, 4, ground_truth_v.shape[1])
    y = np.linspace(0, 4, ground_truth_v.shape[2])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(14,6))
    subfigs = fig.subfigures(1, 2, wspace=0.2)
    axes = subfigs[0].subplots(2,2)
    # fig, axes = plt.subplots(2, 3, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1, 1.2]})
    plt.subplots_adjust(wspace=0.4, hspace=0.3)

    # Plot Ground Truth
    vmax = np.max(np.abs(ground_truth_v[t_idx]))
    vmin = -vmax
    im0 = axes[0, 0].pcolormesh(X, Y, ground_truth_v[t_idx].T, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"Ground Truth V: t={time[t_idx]:.2f}")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0, 0])

    # Plot 3 model predictions
    rows = (0, 1, 1)
    cols = (1, 0, 1)
    for i, model in enumerate(model_order):
        row = rows[i]
        col = cols[i]
        vmax = np.max(np.abs(extrapolated_v[model][t_idx]))
        vmin = -vmax
        im = axes[row, col].pcolormesh(X, Y, extrapolated_v[model][t_idx].T, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row, col].set_title(f"{model} V: t={time[t_idx]:.2f}")
        axes[row, col].set_xlabel("x")
        axes[row, col].set_ylabel("y")
        fig.colorbar(im, ax=axes[row, col])

    # Relative L2 Error plot
    ax_err = subfigs[1].subplots()
    ax_err.grid()
    all_models = ["NeuSA", "PINN", "QRes", "FLS"]
    for model, color in zip(all_models, ['blue', 'red', 'green', 'orange']):
        ax_err.plot(time, relative_l2_errors[model], label=model, color=color)
    ax_err.set_title("Relative L2 Error over Time")
    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("Relative L2 Error")
    ax_err.axvline(1.0, color='black', linestyle='--')
    ax_err.set_xlim(time[0], time[-1])
    max_err = max([np.max(relative_l2_errors[key]) for key in all_models])
    ax_err.set_ylim(0, max_err*1.1)
    ax_err.fill_between(time[time>=1.0], 0, 10, color='gray', alpha=0.2)
    ax_err.text(0.6, 1.05, "Training region", ha='center', size="large")
    ax_err.text(1.4, 1.05, "Extrapolation region", ha='center', size="large")
    ax_err.legend()


    plt.tight_layout()
    return fig