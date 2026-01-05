const std = @import("std");

const vulkan = @cImport({
    @cInclude("vulkan/vulkan.h");
});

// Constants
const DATA_SIZE: usize = 1024; // Number of elements
const BUFFER_SIZE: usize = DATA_SIZE * @sizeOf(f32);
const DESIRED_WORKGROUP_SIZE: u32 = 1024;
const REQUIRED_ALIGNMENT = std.mem.Alignment.fromByteUnits(4096);

const LogicalDeviceInfo = struct {
    handle: vulkan.VkDevice,
    queue: vulkan.VkQueue,
};
const PipelineResources = struct {
    pipeline: vulkan.VkPipeline,
    layout: vulkan.VkPipelineLayout,
    set_layout: vulkan.VkDescriptorSetLayout,
};
const DescriptorResources = struct {
    pool: vulkan.VkDescriptorPool,
    set: vulkan.VkDescriptorSet,
};

fn check(result: vulkan.VkResult, msg: []const u8) !void {
    if (result != vulkan.VK_SUCCESS) {
        std.debug.print("Error: {s} (code: {})\n", .{ msg, result });
        return error.VulkanError;
    }
}

fn createInstance() !vulkan.VkInstance {
    const app_info = vulkan.VkApplicationInfo{
        .sType = vulkan.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = vulkan.VK_API_VERSION_1_4,
        .pApplicationName = "Zig Saxpy",
        .pNext = null,
        .applicationVersion = 0,
        .pEngineName = null,
        .engineVersion = 0,
    };

    const layers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};
    const printf_feature = [_]vulkan.VkValidationFeatureEnableEXT{vulkan.VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};

    var validation_features = vulkan.VkValidationFeaturesEXT{
        .sType = vulkan.VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
        .pNext = null,
        .enabledValidationFeatureCount = printf_feature.len,
        .pEnabledValidationFeatures = &printf_feature,
        .disabledValidationFeatureCount = 0,
        .pDisabledValidationFeatures = null,
    };

    var instance: vulkan.VkInstance = undefined;
    try check(vulkan.vkCreateInstance(&.{
        .sType = vulkan.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = &validation_features,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = layers.len,
        .ppEnabledLayerNames = &layers,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
        .flags = 0,
    }, null, &instance), "CreateInstance");

    return instance;
}

fn pickPhysicalDevice(instance: vulkan.VkInstance, allocator: std.mem.Allocator) !PhysicalDeviceInfo {
    var device_count: u32 = 0;
    try check(vulkan.vkEnumeratePhysicalDevices(instance, &device_count, null), "EnumPhysicalDevices");
    const p_devices = try allocator.alloc(vulkan.VkPhysicalDevice, device_count);
    defer allocator.free(p_devices);
    try check(vulkan.vkEnumeratePhysicalDevices(instance, &device_count, p_devices.ptr), "EnumPhysicalDevices");

    const physical_device = p_devices[0];

    var properties: vulkan.struct_VkPhysicalDeviceProperties = std.mem.zeroes(vulkan.VkPhysicalDeviceProperties);
    vulkan.vkGetPhysicalDeviceProperties(physical_device, &properties);
    if (properties.apiVersion != 0) {
        std.debug.print("Device Name: {s}\n", .{std.mem.sliceTo(&properties.deviceName, 0)});
    }

    const max_dispatch_x = properties.limits.maxComputeWorkGroupSize[0];
    const max_invocations = properties.limits.maxComputeWorkGroupInvocations;
    const actual_workgroup_size: u32 = @min(DESIRED_WORKGROUP_SIZE, @min(max_dispatch_x, max_invocations));

    var queue_family_count: u32 = 0;
    vulkan.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, null);
    const queue_props = try allocator.alloc(vulkan.VkQueueFamilyProperties, queue_family_count);
    defer allocator.free(queue_props);
    vulkan.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_props.ptr);

    var queue_family_index: u32 = 0;
    var found_queue = false;
    for (queue_props, 0..) |prop, idx| {
        if ((prop.queueFlags & vulkan.VK_QUEUE_COMPUTE_BIT) != 0) {
            queue_family_index = @intCast(idx);
            found_queue = true;
            break;
        }
    }

    if (!found_queue) {
        return error.NoComputeQueue;
    }

    return PhysicalDeviceInfo{
        .handle = physical_device,
        .queue_family_index = queue_family_index,
        .workgroup_size = actual_workgroup_size,
    };
}

fn createLogicalDevice(physical_device: vulkan.VkPhysicalDevice, queue_family_index: u32) !LogicalDeviceInfo {
    const queue_priority: f32 = 1.0;
    const queue_create_info = vulkan.VkDeviceQueueCreateInfo{
        .sType = vulkan.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = queue_family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
        .flags = 0,
        .pNext = null,
    };

    var device: vulkan.VkDevice = undefined;
    try check(vulkan.vkCreateDevice(physical_device, &.{
        .sType = vulkan.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_create_info,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null,
        .pEnabledFeatures = null,
        .flags = 0,
        .pNext = null,
    }, null, &device), "CreateDevice");

    var queue: vulkan.VkQueue = undefined;
    vulkan.vkGetDeviceQueue(device, queue_family_index, 0, &queue);

    return LogicalDeviceInfo{ .handle = device, .queue = queue };
}

fn createCommandPool(device: vulkan.VkDevice, queue_family_index: u32) !vulkan.VkCommandPool {
    var command_pool: vulkan.VkCommandPool = undefined;
    try check(vulkan.vkCreateCommandPool(device, &.{
        .sType = vulkan.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = queue_family_index,
        .flags = 0,
        .pNext = null,
    }, null, &command_pool), "CmdPool");
    return command_pool;
}

fn createBuffer(p_dev: vulkan.VkPhysicalDevice, dev: vulkan.VkDevice, size: vulkan.VkDeviceSize, usage: vulkan.VkBufferUsageFlags, properties: vulkan.VkMemoryPropertyFlags, buf: *vulkan.VkBuffer, mem: *vulkan.VkDeviceMemory) !void {
    try check(vulkan.vkCreateBuffer(dev, &.{
        .sType = vulkan.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = vulkan.VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = null,
        .flags = 0,
        .pNext = null,
    }, null, buf), "CreateBuffer");

    var mem_reqs: vulkan.VkMemoryRequirements = undefined;
    vulkan.vkGetBufferMemoryRequirements(dev, buf.*, &mem_reqs);

    const memory_type_index = try findMemoryType(p_dev, mem_reqs.memoryTypeBits, properties);

    try check(vulkan.vkAllocateMemory(dev, &.{
        .sType = vulkan.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = memory_type_index,
        .pNext = null,
    }, null, mem), "AllocateMemory");

    try check(vulkan.vkBindBufferMemory(dev, buf.*, mem.*, 0), "BindBufferMemory");
}

fn findMemoryType(p_dev: vulkan.VkPhysicalDevice, type_filter: u32, properties: vulkan.VkMemoryPropertyFlags) !u32 {
    var mem_props: vulkan.VkPhysicalDeviceMemoryProperties = undefined;
    vulkan.vkGetPhysicalDeviceMemoryProperties(p_dev, &mem_props);

    for (0..mem_props.memoryTypeCount) |i| {
        if ((type_filter & (@as(u32, 1) << @as(u5, @intCast(i)))) != 0 and
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) return @intCast(i);
    }
    return error.NoSuitableMemory;
}

fn createComputePipeline(spirv_file_path: []const u8, device: vulkan.VkDevice, workgroup_size: u32) !PipelineResources {
    const bindings = [_]vulkan.VkDescriptorSetLayoutBinding{
        .{ .binding = 0, .descriptorType = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = vulkan.VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = null },
        .{ .binding = 1, .descriptorType = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = vulkan.VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = null },
    };

    var set_layout: vulkan.VkDescriptorSetLayout = undefined;
    try check(vulkan.vkCreateDescriptorSetLayout(device, &.{
        .sType = vulkan.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = bindings.len,
        .pBindings = &bindings,
        .flags = 0,
        .pNext = null,
    }, null, &set_layout), "DescLayout");

    const push_constant_range = vulkan.VkPushConstantRange{
        .stageFlags = vulkan.VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = @sizeOf(f32) + @sizeOf(u32),
    };

    var pipeline_layout: vulkan.VkPipelineLayout = undefined;
    if (vulkan.vkCreatePipelineLayout(device, &.{
        .sType = vulkan.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range,
        .flags = 0,
        .pNext = null,
    }, null, &pipeline_layout) != vulkan.VK_SUCCESS) {
        vulkan.vkDestroyDescriptorSetLayout(device, set_layout, null);
        return error.VulkanError;
    }

    const file = try std.fs.cwd().openFile(spirv_file_path, .{});
    defer file.close();
    const file_size = try file.getEndPos();
    const mapped_code = try std.posix.mmap(null, file_size, std.posix.PROT.READ, .{ .TYPE = .PRIVATE }, file.handle, 0);
    defer std.posix.munmap(mapped_code);

    var shader_module: vulkan.VkShaderModule = undefined;
    if (vulkan.vkCreateShaderModule(device, &.{
        .sType = vulkan.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = file_size,
        .pCode = @ptrCast(@alignCast(mapped_code.ptr)),
        .flags = 0,
        .pNext = null,
    }, null, &shader_module) != vulkan.VK_SUCCESS) {
        vulkan.vkDestroyPipelineLayout(device, pipeline_layout, null);
        vulkan.vkDestroyDescriptorSetLayout(device, set_layout, null);
        return error.VulkanError;
    }
    defer vulkan.vkDestroyShaderModule(device, shader_module, null);

    const spec_map = [_]vulkan.VkSpecializationMapEntry{.{ .constantID = 0, .offset = 0, .size = @sizeOf(u32) }};
    const spec_data = [_]u32{workgroup_size};
    const spec_info = vulkan.VkSpecializationInfo{ .mapEntryCount = spec_map.len, .pMapEntries = &spec_map, .dataSize = @sizeOf(@TypeOf(spec_data)), .pData = &spec_data };

    var compute_pipeline: vulkan.VkPipeline = undefined;
    const pipeline_check = vulkan.vkCreateComputePipelines(device, null, 1, &.{ .sType = vulkan.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = .{
        .sType = vulkan.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = vulkan.VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
        .flags = 0,
        .pNext = null,
        .pSpecializationInfo = &spec_info,
    }, .layout = pipeline_layout, .flags = 0, .pNext = null, .basePipelineHandle = null, .basePipelineIndex = 0 }, null, &compute_pipeline);

    if (pipeline_check != vulkan.VK_SUCCESS) {
        vulkan.vkDestroyPipelineLayout(device, pipeline_layout, null);
        vulkan.vkDestroyDescriptorSetLayout(device, set_layout, null);
        return error.VulkanError;
    }

    return PipelineResources{ .pipeline = compute_pipeline, .layout = pipeline_layout, .set_layout = set_layout };
}

fn createDescriptorSets(device: vulkan.VkDevice, set_layout: vulkan.VkDescriptorSetLayout, buffer_x: vulkan.VkBuffer, buffer_y: vulkan.VkBuffer) !DescriptorResources {
    const pool_sizes = [_]vulkan.VkDescriptorPoolSize{.{ .type = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 }};
    var desc_pool: vulkan.VkDescriptorPool = undefined;
    try check(vulkan.vkCreateDescriptorPool(device, &.{
        .sType = vulkan.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_sizes,
        .flags = 0,
        .pNext = null,
    }, null, &desc_pool), "DescPool");

    var desc_set: vulkan.VkDescriptorSet = undefined;
    if (vulkan.vkAllocateDescriptorSets(device, &.{
        .sType = vulkan.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &set_layout,
        .pNext = null,
    }, &desc_set) != vulkan.VK_SUCCESS) {
        vulkan.vkDestroyDescriptorPool(device, desc_pool, null);
        return error.VulkanError;
    }

    const buffer_size_bytes: vulkan.VkDeviceSize = BUFFER_SIZE;
    const buffer_info_x = vulkan.VkDescriptorBufferInfo{ .buffer = buffer_x, .offset = 0, .range = buffer_size_bytes };
    const buffer_info_y = vulkan.VkDescriptorBufferInfo{ .buffer = buffer_y, .offset = 0, .range = buffer_size_bytes };
    const writes = [_]vulkan.VkWriteDescriptorSet{
        .{ .sType = vulkan.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buffer_info_x, .pImageInfo = null, .pTexelBufferView = null, .dstArrayElement = 0, .pNext = null },
        .{ .sType = vulkan.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buffer_info_y, .pImageInfo = null, .pTexelBufferView = null, .dstArrayElement = 0, .pNext = null },
    };
    vulkan.vkUpdateDescriptorSets(device, writes.len, &writes, 0, null);

    return DescriptorResources{ .pool = desc_pool, .set = desc_set };
}

fn recordComputeCommands(cmd: vulkan.VkCommandBuffer, pipeline: vulkan.VkPipeline, layout: vulkan.VkPipelineLayout, desc_set: vulkan.VkDescriptorSet, staging_buffer_x: vulkan.VkBuffer, staging_buffer_y: vulkan.VkBuffer, device_buffer_x: vulkan.VkBuffer, device_buffer_y: vulkan.VkBuffer, buffer_size: vulkan.VkDeviceSize, workgroup_size: u32, data_count: u32) !void {
    const copy_region = vulkan.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = buffer_size };

    try check(vulkan.vkBeginCommandBuffer(cmd, &.{
        .sType = vulkan.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = vulkan.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = null,
        .pNext = null,
    }), "BeginCmd");

    vulkan.vkCmdCopyBuffer(cmd, staging_buffer_x, device_buffer_x, 1, &copy_region);
    vulkan.vkCmdCopyBuffer(cmd, staging_buffer_y, device_buffer_y, 1, &copy_region);

    const barrier_upload = [_]vulkan.VkBufferMemoryBarrier{
        .{ .sType = vulkan.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, .srcAccessMask = vulkan.VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = vulkan.VK_ACCESS_SHADER_READ_BIT | vulkan.VK_ACCESS_SHADER_WRITE_BIT, .srcQueueFamilyIndex = vulkan.VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = vulkan.VK_QUEUE_FAMILY_IGNORED, .buffer = device_buffer_x, .offset = 0, .size = buffer_size, .pNext = null },
        .{ .sType = vulkan.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, .srcAccessMask = vulkan.VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = vulkan.VK_ACCESS_SHADER_READ_BIT | vulkan.VK_ACCESS_SHADER_WRITE_BIT, .srcQueueFamilyIndex = vulkan.VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = vulkan.VK_QUEUE_FAMILY_IGNORED, .buffer = device_buffer_y, .offset = 0, .size = buffer_size, .pNext = null },
    };
    vulkan.vkCmdPipelineBarrier(cmd, vulkan.VK_PIPELINE_STAGE_TRANSFER_BIT, vulkan.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, null, barrier_upload.len, &barrier_upload, 0, null);

    vulkan.vkCmdBindPipeline(cmd, vulkan.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vulkan.vkCmdBindDescriptorSets(cmd, vulkan.VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &desc_set, 0, null);

    const push_values = struct { a: f32, n: u32 }{ .a = 2.5, .n = data_count };
    vulkan.vkCmdPushConstants(cmd, layout, vulkan.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(@TypeOf(push_values)), &push_values);

    const groups = (data_count + workgroup_size - 1) / workgroup_size;
    vulkan.vkCmdDispatch(cmd, groups, 1, 1);

    const barrier_download = vulkan.VkBufferMemoryBarrier{
        .sType = vulkan.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = vulkan.VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = vulkan.VK_ACCESS_TRANSFER_READ_BIT,
        .srcQueueFamilyIndex = vulkan.VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = vulkan.VK_QUEUE_FAMILY_IGNORED,
        .buffer = device_buffer_y,
        .offset = 0,
        .size = buffer_size,
        .pNext = null,
    };
    vulkan.vkCmdPipelineBarrier(cmd, vulkan.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vulkan.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, null, 1, &barrier_download, 0, null);

    vulkan.vkCmdCopyBuffer(cmd, device_buffer_y, staging_buffer_y, 1, &copy_region);

    try check(vulkan.vkEndCommandBuffer(cmd), "EndCmd");
}

const PhysicalDeviceInfo = struct {
    handle: vulkan.VkPhysicalDevice,
    queue_family_index: u32,
    workgroup_size: u32,
};

const VulkanContext = struct {
    instance: vulkan.VkInstance,
    physical_device: vulkan.VkPhysicalDevice,
    device: vulkan.VkDevice,
    queue: vulkan.VkQueue,
    command_pool: vulkan.VkCommandPool,
    queue_family_index: u32,
    workgroup_size: u32,
    vk_allocator: ?*const vulkan.VkAllocationCallbacks = null,

    pub fn init(allocator: std.mem.Allocator) !VulkanContext {
        std.debug.print("Initializing Vulkan Context...\n", .{});
        const instance = try createInstance();
        errdefer vulkan.vkDestroyInstance(instance, null);

        const physical_info = try pickPhysicalDevice(instance, allocator);
        std.debug.print("Using Workgroup Size: {}\n", .{physical_info.workgroup_size});

        const logical_info = try createLogicalDevice(physical_info.handle, physical_info.queue_family_index);
        errdefer vulkan.vkDestroyDevice(logical_info.handle, null);

        const cmd_pool = try createCommandPool(logical_info.handle, physical_info.queue_family_index);

        return VulkanContext{
            .instance = instance,
            .physical_device = physical_info.handle,
            .device = logical_info.handle,
            .queue = logical_info.queue,
            .command_pool = cmd_pool,
            .queue_family_index = physical_info.queue_family_index,
            .workgroup_size = physical_info.workgroup_size,
        };
    }

    pub fn deinit(self: *VulkanContext) void {
        vulkan.vkDestroyCommandPool(self.device, self.command_pool, self.vk_allocator);
        vulkan.vkDestroyDevice(self.device, self.vk_allocator);
        vulkan.vkDestroyInstance(self.instance, self.vk_allocator);
    }
};

pub const SaxpyApp = struct {
    ctx: VulkanContext,

    device_buffer_x: vulkan.VkBuffer,
    device_memory_x: vulkan.VkDeviceMemory,
    device_buffer_y: vulkan.VkBuffer,
    device_memory_y: vulkan.VkDeviceMemory,
    staging_buffer_x: vulkan.VkBuffer,
    staging_memory_x: vulkan.VkDeviceMemory,
    staging_buffer_y: vulkan.VkBuffer,
    staging_memory_y: vulkan.VkDeviceMemory,

    pipeline: vulkan.VkPipeline,
    pipeline_layout: vulkan.VkPipelineLayout,
    descriptor_set_layout: vulkan.VkDescriptorSetLayout,
    descriptor_pool: vulkan.VkDescriptorPool,
    descriptor_set: vulkan.VkDescriptorSet,

    command_buffer: vulkan.VkCommandBuffer,

    pub fn init(allocator: std.mem.Allocator) !SaxpyApp {
        var ctx = try VulkanContext.init(allocator);
        errdefer ctx.deinit();

        const buffer_size_bytes: vulkan.VkDeviceSize = BUFFER_SIZE;

        const device_usage = vulkan.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vulkan.VK_BUFFER_USAGE_TRANSFER_DST_BIT | vulkan.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        const device_props = vulkan.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        var db_x: vulkan.VkBuffer = undefined;
        var dm_x: vulkan.VkDeviceMemory = undefined;
        try createBuffer(ctx.physical_device, ctx.device, buffer_size_bytes, device_usage, device_props, &db_x, &dm_x);
        errdefer {
            vulkan.vkDestroyBuffer(ctx.device, db_x, null);
            vulkan.vkFreeMemory(ctx.device, dm_x, null);
        }

        var db_y: vulkan.VkBuffer = undefined;
        var dm_y: vulkan.VkDeviceMemory = undefined;
        try createBuffer(ctx.physical_device, ctx.device, buffer_size_bytes, device_usage, device_props, &db_y, &dm_y);
        errdefer {
            vulkan.vkDestroyBuffer(ctx.device, db_y, null);
            vulkan.vkFreeMemory(ctx.device, dm_y, null);
        }

        const staging_usage = vulkan.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vulkan.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        const staging_props = vulkan.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vulkan.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        var sb_x: vulkan.VkBuffer = undefined;
        var sm_x: vulkan.VkDeviceMemory = undefined;
        try createBuffer(ctx.physical_device, ctx.device, buffer_size_bytes, staging_usage, staging_props, &sb_x, &sm_x);
        errdefer {
            vulkan.vkDestroyBuffer(ctx.device, sb_x, null);
            vulkan.vkFreeMemory(ctx.device, sm_x, null);
        }

        var sb_y: vulkan.VkBuffer = undefined;
        var sm_y: vulkan.VkDeviceMemory = undefined;
        try createBuffer(ctx.physical_device, ctx.device, buffer_size_bytes, staging_usage, staging_props, &sb_y, &sm_y);
        errdefer {
            vulkan.vkDestroyBuffer(ctx.device, sb_y, null);
            vulkan.vkFreeMemory(ctx.device, sm_y, null);
        }

        const pipe_res = try createComputePipeline("saxpy.spv", ctx.device, ctx.workgroup_size);
        errdefer {
            vulkan.vkDestroyPipeline(ctx.device, pipe_res.pipeline, null);
            vulkan.vkDestroyPipelineLayout(ctx.device, pipe_res.layout, null);
            vulkan.vkDestroyDescriptorSetLayout(ctx.device, pipe_res.set_layout, null);
        }

        const desc_res = try createDescriptorSets(ctx.device, pipe_res.set_layout, db_x, db_y);
        errdefer vulkan.vkDestroyDescriptorPool(ctx.device, desc_res.pool, null);

        var cmd_buf: vulkan.VkCommandBuffer = undefined;
        try check(vulkan.vkAllocateCommandBuffers(ctx.device, &.{
            .sType = vulkan.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = ctx.command_pool,
            .level = vulkan.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
            .pNext = null,
        }, &cmd_buf), "AllocCmdBuf");

        return SaxpyApp{
            .ctx = ctx,
            .device_buffer_x = db_x,
            .device_memory_x = dm_x,
            .device_buffer_y = db_y,
            .device_memory_y = dm_y,
            .staging_buffer_x = sb_x,
            .staging_memory_x = sm_x,
            .staging_buffer_y = sb_y,
            .staging_memory_y = sm_y,
            .pipeline = pipe_res.pipeline,
            .pipeline_layout = pipe_res.layout,
            .descriptor_set_layout = pipe_res.set_layout,
            .descriptor_pool = desc_res.pool,
            .descriptor_set = desc_res.set,
            .command_buffer = cmd_buf,
        };
    }

    pub fn deinit(self: *SaxpyApp) void {
        const d = self.ctx.device;
        const a = self.ctx.vk_allocator;

        vulkan.vkDestroyDescriptorPool(d, self.descriptor_pool, a);
        vulkan.vkDestroyPipeline(d, self.pipeline, a);
        vulkan.vkDestroyPipelineLayout(d, self.pipeline_layout, a);
        vulkan.vkDestroyDescriptorSetLayout(d, self.descriptor_set_layout, a);

        vulkan.vkDestroyBuffer(d, self.staging_buffer_y, a);
        vulkan.vkFreeMemory(d, self.staging_memory_y, a);
        vulkan.vkDestroyBuffer(d, self.staging_buffer_x, a);
        vulkan.vkFreeMemory(d, self.staging_memory_x, a);
        vulkan.vkDestroyBuffer(d, self.device_buffer_y, a);
        vulkan.vkFreeMemory(d, self.device_memory_y, a);
        vulkan.vkDestroyBuffer(d, self.device_buffer_x, a);
        vulkan.vkFreeMemory(d, self.device_memory_x, a);

        self.ctx.deinit();
    }

    pub fn run(self: *SaxpyApp) !void {
        const device = self.ctx.device;
        const buffer_size_bytes = BUFFER_SIZE;

        var host_data_x: [*]f32 = undefined;
        try check(vulkan.vkMapMemory(device, self.staging_memory_x, 0, buffer_size_bytes, 0, @ptrCast(&host_data_x)), "Map Staging X");
        var host_data_y: [*]f32 = undefined;
        try check(vulkan.vkMapMemory(device, self.staging_memory_y, 0, buffer_size_bytes, 0, @ptrCast(&host_data_y)), "Map Staging Y");

        for (0..DATA_SIZE) |idx| {
            host_data_x[idx] = @floatFromInt(idx);
            host_data_y[idx] = 100.0;
        }

        vulkan.vkUnmapMemory(device, self.staging_memory_x);
        vulkan.vkUnmapMemory(device, self.staging_memory_y);

        try recordComputeCommands(
            self.command_buffer,
            self.pipeline,
            self.pipeline_layout,
            self.descriptor_set,
            self.staging_buffer_x,
            self.staging_buffer_y,
            self.device_buffer_x,
            self.device_buffer_y,
            buffer_size_bytes,
            self.ctx.workgroup_size,
            DATA_SIZE,
        );

        std.debug.print("Submitting compute job...\n", .{});
        const submit_info = vulkan.VkSubmitInfo{
            .sType = vulkan.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &self.command_buffer,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
            .pNext = null,
        };
        try check(vulkan.vkQueueSubmit(self.ctx.queue, 1, &submit_info, null), "QueueSubmit");
        try check(vulkan.vkQueueWaitIdle(self.ctx.queue), "WaitIdle");
    }

    pub fn verify(self: *SaxpyApp) !void {
        const buffer_size_bytes = BUFFER_SIZE;
        var host_data_y: [*]f32 = undefined;
        try check(vulkan.vkMapMemory(self.ctx.device, self.staging_memory_y, 0, buffer_size_bytes, 0, @ptrCast(&host_data_y)), "Map Staging Y Read");

        std.debug.print("Verification (First 5 elements):\n", .{});
        std.debug.print("Formula: Y = 2.5 * X + Y\n", .{});
        for (0..5) |i| {
            const x_val: f32 = @floatFromInt(i);
            const expected = 2.5 * x_val + 100.0;
            std.debug.print("Index {}: Got {d:.2}, Expected {d:.2}\n", .{ i, host_data_y[i], expected });
        }
        vulkan.vkUnmapMemory(self.ctx.device, self.staging_memory_y);
    }
};
