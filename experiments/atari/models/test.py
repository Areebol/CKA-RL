import torch
import torch.nn as nn
from torch.optim import Adam

def update_theta(theta_old, tau_list, alpha, Delta_theta):
    # 将theta_old的参数提取为可更新的参数形式（假设theta_old是一个nn.Module实例）
    theta_old_params = list(theta_old.parameters())
    # 处理tau中的模块参数
    tau_params_sum = None
    for a, tau_module in zip(alpha, tau_list):
        tau_params = list(tau_module.parameters())
        if tau_params_sum is None:
            tau_params_sum = [a * p.clone() for p in tau_params]
        else:
            for i in range(len(tau_params)):
                tau_params_sum[i] += a * tau_params[i]

    # 将Delta_theta.parameters()生成器转换为列表，方便后续索引操作
    Delta_theta_params = list(Delta_theta.parameters())

    # 加权融合tau的参数总和与theta_old的参数
    for i in range(len(theta_old_params)):
        theta_old_params[i].data += tau_params_sum[i].data + Delta_theta_params[i].data

    return theta_old


# 假设theta_old是一个简单的线性层作为示例
theta_old = nn.Linear(10, 5)
# 从pt文件读取的多个module（这里简单模拟创建几个相同结构的线性层作为示例）
tau_modules = [nn.Linear(10, 5) for _ in range(3)]
# 可学习的加权系数alpha（初始化为一个可训练的参数）
alpha = nn.Parameter(torch.ones(3))
# 可学习的Delta_theta（也是一个线性层示例）
Delta_theta = nn.Linear(10, 5)

# 定义优化器来更新alpha和Delta_theta
optimizer = Adam([alpha, *Delta_theta.parameters()], lr=0.001)

# 虚拟的输入数据（batch_size=4作为示例）
input_data = torch.rand(4, 10)
# 虚拟的目标输出（简单示例，真实场景按实际任务定）
target = torch.rand(4, 5)

for _ in range(10):  # 简单训练几次的循环示例
    # 先更新theta_new（也就是更新theta_old的参数）
    theta_old = update_theta(theta_old, tau_modules, alpha, Delta_theta)

    # 前向传播得到输出
    output = theta_old(input_data)
    # 定义损失函数（这里使用均方误差示例）
    loss = nn.MSELoss()(output, target)
    # 反向传播计算梯度并更新alpha和Delta_theta
    optimizer.zero_grad()
    loss.backward()
    print(alpha.grad, Delta_theta.weight.grad, Delta_theta.bias.grad, theta_old.weight.grad)
    optimizer.step()