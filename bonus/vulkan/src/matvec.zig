const std = @import("std");

const vulkan = @cImport({
    @cInclude("vulkan/vulkan.h");
});

// Constants
const ROWS: u32 = 4096;
const COLS: u32 = 4096;
const VEC_SIZE_BYTES: usize = COLS * @sizeOf(f32);
const MAT_SIZE_BYTES: usize = ROWS * COLS * @sizeOf(f32);
const OUT_SIZE_BYTES: usize = ROWS * @sizeOf(f32);
const DESIRED_WORKGROUP_SIZE: u32 = 256; // Must match shader layout(local_size_x_id = 1) if specialized, or just be efficient

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

pub fn loadPipelineCache(device: vulkan.VkDevice, cache_path: []const u8, allocator: std.mem.Allocator) !vulkan.VkPipelineCache {
    var initial_data: []u8 = &.{};

    if (std.fs.cwd().openFile(cache_path, .{})) |file| {
        defer file.close();
        initial_data = try file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    } else |_| {
        std.debug.print("No valid pipeline cache found at {s}, creating new one.\n", .{cache_path});
    }
    defer if (initial_data.len > 0) allocator.free(initial_data);

    var cache: vulkan.VkPipelineCache = undefined;
    const create_info = vulkan.VkPipelineCacheCreateInfo{
        .sType = vulkan.VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .initialDataSize = initial_data.len,
        .pInitialData = if (initial_data.len > 0) initial_data.ptr else null,
    };

    if (vulkan.vkCreatePipelineCache(device, &create_info, null, &cache) != vulkan.VK_SUCCESS) {
        return error.VulkanError;
    }

    return cache;
}

pub fn savePipelineCache(device: vulkan.VkDevice, cache: vulkan.VkPipelineCache, cache_path: []const u8, allocator: std.mem.Allocator) !void {
    var size: usize = 0;
    if (vulkan.vkGetPipelineCacheData(device, cache, &size, null) != vulkan.VK_SUCCESS) {
        return error.VulkanError;
    }

    const data = try allocator.alloc(u8, size);
    defer allocator.free(data);

    if (vulkan.vkGetPipelineCacheData(device, cache, &size, data.ptr) != vulkan.VK_SUCCESS) {
        return error.VulkanError;
    }

    const file = try std.fs.cwd().createFile(cache_path, .{});
    defer file.close();
    try file.writeAll(data);
    std.debug.print("Pipeline cache saved to {s} ({d} bytes)\n", .{ cache_path, size });
}

fn createInstance() !vulkan.VkInstance {
    const app_info = vulkan.VkApplicationInfo{
        .sType = vulkan.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = vulkan.VK_API_VERSION_1_4,
        .pApplicationName = "Zig Matvec",
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

fn createComputePipeline(spirv_file_path: []const u8, device: vulkan.VkDevice, workgroup_size: u32, pipeline_cache_file_path: []const u8, allocator: std.mem.Allocator) !PipelineResources {
    const pipeline_cache = try loadPipelineCache(device, pipeline_cache_file_path, allocator);
    defer vulkan.vkDestroyPipelineCache(device, pipeline_cache, null);

    const bindings = [_]vulkan.VkDescriptorSetLayoutBinding{
        .{ .binding = 0, .descriptorType = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = vulkan.VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = null },
        .{ .binding = 1, .descriptorType = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = vulkan.VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = null },
        .{ .binding = 2, .descriptorType = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = vulkan.VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = null },
    };

    var set_layout: vulkan.VkDescriptorSetLayout = undefined;
    try check(vulkan.vkCreateDescriptorSetLayout(device, &.{
        .sType = vulkan.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = bindings.len,
        .pBindings = &bindings,
        .flags = 0,
        .pNext = null,
    }, null, &set_layout), "DescLayout");

    // Push Constants for rows, cols
    const push_constant_range = vulkan.VkPushConstantRange{
        .stageFlags = vulkan.VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 2 * @sizeOf(u32),
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

    const spec_map = [_]vulkan.VkSpecializationMapEntry{.{ .constantID = 1, .offset = 0, .size = @sizeOf(u32) }};
    const spec_data = [_]u32{workgroup_size};
    const spec_info = vulkan.VkSpecializationInfo{ .mapEntryCount = spec_map.len, .pMapEntries = &spec_map, .dataSize = @sizeOf(@TypeOf(spec_data)), .pData = &spec_data };

    var compute_pipeline: vulkan.VkPipeline = undefined;

    const pipeline_check = vulkan.vkCreateComputePipelines(device, pipeline_cache, 1, //
        &.{
            .sType = vulkan.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, //
            .stage = .{
                .sType = vulkan.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = vulkan.VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .flags = 0,
                .pNext = null,
                .pSpecializationInfo = &spec_info,
            },
            .layout = pipeline_layout,
            .flags = 0,
            .pNext = null,
            .basePipelineHandle = null,
            .basePipelineIndex = 0,
        }, null, &compute_pipeline);

    if (pipeline_check != vulkan.VK_SUCCESS) {
        vulkan.vkDestroyPipelineLayout(device, pipeline_layout, null);
        vulkan.vkDestroyDescriptorSetLayout(device, set_layout, null);
        return error.VulkanError;
    }
    try savePipelineCache(device, pipeline_cache, pipeline_cache_file_path, allocator);
    return PipelineResources{ .pipeline = compute_pipeline, .layout = pipeline_layout, .set_layout = set_layout };
}

fn createDescriptorSets(device: vulkan.VkDevice, set_layout: vulkan.VkDescriptorSetLayout, buffer_out: vulkan.VkBuffer, buffer_in: vulkan.VkBuffer, buffer_weights: vulkan.VkBuffer) !DescriptorResources {
    const pool_sizes = [_]vulkan.VkDescriptorPoolSize{.{ .type = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3 }};
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

    const info_out = vulkan.VkDescriptorBufferInfo{ .buffer = buffer_out, .offset = 0, .range = OUT_SIZE_BYTES };
    const info_in = vulkan.VkDescriptorBufferInfo{ .buffer = buffer_in, .offset = 0, .range = VEC_SIZE_BYTES };
    const info_weights = vulkan.VkDescriptorBufferInfo{ .buffer = buffer_weights, .offset = 0, .range = MAT_SIZE_BYTES };

    const writes = [_]vulkan.VkWriteDescriptorSet{
        .{ .sType = vulkan.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &info_out, .pImageInfo = null, .pTexelBufferView = null, .dstArrayElement = 0, .pNext = null },
        .{ .sType = vulkan.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &info_in, .pImageInfo = null, .pTexelBufferView = null, .dstArrayElement = 0, .pNext = null },
        .{ .sType = vulkan.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &info_weights, .pImageInfo = null, .pTexelBufferView = null, .dstArrayElement = 0, .pNext = null },
    };
    vulkan.vkUpdateDescriptorSets(device, writes.len, &writes, 0, null);

    return DescriptorResources{ .pool = desc_pool, .set = desc_set };
}

fn recordComputeCommands(cmd: vulkan.VkCommandBuffer, pipeline: vulkan.VkPipeline, layout: vulkan.VkPipelineLayout, desc_set: vulkan.VkDescriptorSet, staging_in: vulkan.VkBuffer, staging_weights: vulkan.VkBuffer, dev_out: vulkan.VkBuffer, dev_in: vulkan.VkBuffer, dev_weights: vulkan.VkBuffer, staging_out: vulkan.VkBuffer, workgroup_size: u32) !void {
    const copy_in = vulkan.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = VEC_SIZE_BYTES };
    const copy_weights = vulkan.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = MAT_SIZE_BYTES };
    const copy_out = vulkan.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = OUT_SIZE_BYTES };

    try check(vulkan.vkBeginCommandBuffer(cmd, &.{
        .sType = vulkan.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = vulkan.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = null,
        .pNext = null,
    }), "BeginCmd");

    vulkan.vkCmdCopyBuffer(cmd, staging_in, dev_in, 1, &copy_in);
    vulkan.vkCmdCopyBuffer(cmd, staging_weights, dev_weights, 1, &copy_weights);

    const barriers = [_]vulkan.VkBufferMemoryBarrier{
        .{ .sType = vulkan.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, .srcAccessMask = vulkan.VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = vulkan.VK_ACCESS_SHADER_READ_BIT, .srcQueueFamilyIndex = vulkan.VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = vulkan.VK_QUEUE_FAMILY_IGNORED, .buffer = dev_in, .offset = 0, .size = VEC_SIZE_BYTES, .pNext = null },
        .{ .sType = vulkan.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, .srcAccessMask = vulkan.VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = vulkan.VK_ACCESS_SHADER_READ_BIT, .srcQueueFamilyIndex = vulkan.VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = vulkan.VK_QUEUE_FAMILY_IGNORED, .buffer = dev_weights, .offset = 0, .size = MAT_SIZE_BYTES, .pNext = null },
    };
    vulkan.vkCmdPipelineBarrier(cmd, vulkan.VK_PIPELINE_STAGE_TRANSFER_BIT, vulkan.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, null, barriers.len, &barriers, 0, null);

    vulkan.vkCmdBindPipeline(cmd, vulkan.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vulkan.vkCmdBindDescriptorSets(cmd, vulkan.VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &desc_set, 0, null);

    const push_vals = struct { rows: u32, cols: u32 }{ .rows = ROWS, .cols = COLS };
    vulkan.vkCmdPushConstants(cmd, layout, vulkan.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(@TypeOf(push_vals)), &push_vals);

    const groups = (ROWS + workgroup_size - 1) / workgroup_size;
    vulkan.vkCmdDispatch(cmd, groups, 1, 1);

    const barrier_out = vulkan.VkBufferMemoryBarrier{
        .sType = vulkan.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = vulkan.VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = vulkan.VK_ACCESS_TRANSFER_READ_BIT,
        .srcQueueFamilyIndex = vulkan.VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = vulkan.VK_QUEUE_FAMILY_IGNORED,
        .buffer = dev_out,
        .offset = 0,
        .size = OUT_SIZE_BYTES,
        .pNext = null,
    };
    vulkan.vkCmdPipelineBarrier(cmd, vulkan.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vulkan.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, null, 1, &barrier_out, 0, null);

    vulkan.vkCmdCopyBuffer(cmd, dev_out, staging_out, 1, &copy_out);

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
        std.debug.print("Initializing Vulkan Context for MatVec...\n", .{});
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

pub const MatvecApp = struct {
    ctx: VulkanContext,

    dev_out: vulkan.VkBuffer,
    dev_mem_out: vulkan.VkDeviceMemory,
    dev_in: vulkan.VkBuffer,
    dev_mem_in: vulkan.VkDeviceMemory,
    dev_weights: vulkan.VkBuffer,
    dev_mem_weights: vulkan.VkDeviceMemory,

    staging_out: vulkan.VkBuffer,
    staging_mem_out: vulkan.VkDeviceMemory,
    staging_in: vulkan.VkBuffer,
    staging_mem_in: vulkan.VkDeviceMemory,
    staging_weights: vulkan.VkBuffer,
    staging_mem_weights: vulkan.VkDeviceMemory,

    pipeline: vulkan.VkPipeline,
    pipeline_layout: vulkan.VkPipelineLayout,
    descriptor_set_layout: vulkan.VkDescriptorSetLayout,
    descriptor_pool: vulkan.VkDescriptorPool,
    descriptor_set: vulkan.VkDescriptorSet,

    command_buffer: vulkan.VkCommandBuffer,

    pub fn init(allocator: std.mem.Allocator) !MatvecApp {
        var ctx = try VulkanContext.init(allocator);
        errdefer ctx.deinit();

        const dev_usage = vulkan.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vulkan.VK_BUFFER_USAGE_TRANSFER_DST_BIT | vulkan.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        const dev_props = vulkan.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        var do: vulkan.VkBuffer = undefined;
        var dmo: vulkan.VkDeviceMemory = undefined;
        try createBuffer(ctx.physical_device, ctx.device, OUT_SIZE_BYTES, dev_usage, dev_props, &do, &dmo);

        var di: vulkan.VkBuffer = undefined;
        var dmi: vulkan.VkDeviceMemory = undefined;
        try createBuffer(ctx.physical_device, ctx.device, VEC_SIZE_BYTES, dev_usage, dev_props, &di, &dmi);

        var dw: vulkan.VkBuffer = undefined;
        var dmw: vulkan.VkDeviceMemory = undefined;
        try createBuffer(ctx.physical_device, ctx.device, MAT_SIZE_BYTES, dev_usage, dev_props, &dw, &dmw);

        const staging_usage = vulkan.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vulkan.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        const staging_props = vulkan.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vulkan.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        var so: vulkan.VkBuffer = undefined;
        var smo: vulkan.VkDeviceMemory = undefined;
        try createBuffer(ctx.physical_device, ctx.device, OUT_SIZE_BYTES, staging_usage, staging_props, &so, &smo);

        var si: vulkan.VkBuffer = undefined;
        var smi: vulkan.VkDeviceMemory = undefined;
        try createBuffer(ctx.physical_device, ctx.device, VEC_SIZE_BYTES, staging_usage, staging_props, &si, &smi);

        var sw: vulkan.VkBuffer = undefined;
        var smw: vulkan.VkDeviceMemory = undefined;
        try createBuffer(ctx.physical_device, ctx.device, MAT_SIZE_BYTES, staging_usage, staging_props, &sw, &smw);

        const pipe_res = try createComputePipeline("matvec.spv", ctx.device, ctx.workgroup_size, "app_pipeline_cache.bin", allocator);

        const desc_res = try createDescriptorSets(ctx.device, pipe_res.set_layout, do, di, dw);

        var cmd_buf: vulkan.VkCommandBuffer = undefined;
        try check(vulkan.vkAllocateCommandBuffers(ctx.device, &.{
            .sType = vulkan.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = ctx.command_pool,
            .level = vulkan.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
            .pNext = null,
        }, &cmd_buf), "AllocCmdBuf");

        return MatvecApp{
            .ctx = ctx,
            .dev_out = do,
            .dev_mem_out = dmo,
            .dev_in = di,
            .dev_mem_in = dmi,
            .dev_weights = dw,
            .dev_mem_weights = dmw,
            .staging_out = so,
            .staging_mem_out = smo,
            .staging_in = si,
            .staging_mem_in = smi,
            .staging_weights = sw,
            .staging_mem_weights = smw,
            .pipeline = pipe_res.pipeline,
            .pipeline_layout = pipe_res.layout,
            .descriptor_set_layout = pipe_res.set_layout,
            .descriptor_pool = desc_res.pool,
            .descriptor_set = desc_res.set,
            .command_buffer = cmd_buf,
        };
    }

    pub fn deinit(self: *MatvecApp) void {
        const d = self.ctx.device;
        const a = self.ctx.vk_allocator;

        vulkan.vkDestroyDescriptorPool(d, self.descriptor_pool, a);
        vulkan.vkDestroyPipeline(d, self.pipeline, a);
        vulkan.vkDestroyPipelineLayout(d, self.pipeline_layout, a);
        vulkan.vkDestroyDescriptorSetLayout(d, self.descriptor_set_layout, a);

        vulkan.vkDestroyBuffer(d, self.staging_weights, a);
        vulkan.vkFreeMemory(d, self.staging_mem_weights, a);
        vulkan.vkDestroyBuffer(d, self.staging_in, a);
        vulkan.vkFreeMemory(d, self.staging_mem_in, a);
        vulkan.vkDestroyBuffer(d, self.staging_out, a);
        vulkan.vkFreeMemory(d, self.staging_mem_out, a);

        vulkan.vkDestroyBuffer(d, self.dev_weights, a);
        vulkan.vkFreeMemory(d, self.dev_mem_weights, a);
        vulkan.vkDestroyBuffer(d, self.dev_in, a);
        vulkan.vkFreeMemory(d, self.dev_mem_in, a);
        vulkan.vkDestroyBuffer(d, self.dev_out, a);
        vulkan.vkFreeMemory(d, self.dev_mem_out, a);

        self.ctx.deinit();
    }

    pub fn run(self: *MatvecApp) !void {
        const device = self.ctx.device;

        // Init
        var ptr_in: [*]f32 = undefined;
        try check(vulkan.vkMapMemory(device, self.staging_mem_in, 0, VEC_SIZE_BYTES, 0, @ptrCast(&ptr_in)), "Map In");
        var ptr_weights: [*]f32 = undefined;
        try check(vulkan.vkMapMemory(device, self.staging_mem_weights, 0, MAT_SIZE_BYTES, 0, @ptrCast(&ptr_weights)), "Map Weights");

        // Fill Input Vector
        for (0..COLS) |idx| {
            ptr_in[idx] = @as(f32, @floatFromInt(idx)) * 0.001;
        }

        // Verification
        var idx: usize = 0;
        for (0..ROWS) |_| {
            for (0..COLS) |_| {
                ptr_weights[idx] = 1.0;
                idx += 1;
            }
        }

        vulkan.vkUnmapMemory(device, self.staging_mem_in);
        vulkan.vkUnmapMemory(device, self.staging_mem_weights);

        try recordComputeCommands(self.command_buffer, self.pipeline, self.pipeline_layout, self.descriptor_set, self.staging_in, self.staging_weights, self.dev_out, self.dev_in, self.dev_weights, self.staging_out, self.ctx.workgroup_size);

        std.debug.print("Submitting MatVec job...\n", .{});
        const start = std.time.nanoTimestamp();

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

        const end = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
        const throughput_gflops = (2.0 * @as(f64, @floatFromInt(ROWS * COLS))) / (duration_ms * 1_000_000.0);
        std.debug.print("MatVec completed in {d:.4} ms ({d:.2} GFLOPS)\n", .{ duration_ms, throughput_gflops });
    }

    pub fn verify(self: *MatvecApp) !void {
        var ptr_out: [*]f32 = undefined;
        try check(vulkan.vkMapMemory(self.ctx.device, self.staging_mem_out, 0, OUT_SIZE_BYTES, 0, @ptrCast(&ptr_out)), "Map Out");

        std.debug.print("Verification (First 5 rows):\n", .{});

        const sum_indices = @as(f64, @floatFromInt((COLS - 1) * COLS)) / 2.0;
        const expected_val = @as(f32, @floatCast(sum_indices * 0.001));

        for (0..5) |row| {
            const got = ptr_out[row];
            std.debug.print("Row {}: Got {d:.2}, Expected {d:.2}, Diff {d:.4}\n", .{ row, got, expected_val, @abs(got - expected_val) });
        }
        vulkan.vkUnmapMemory(self.ctx.device, self.staging_mem_out);
    }
};
