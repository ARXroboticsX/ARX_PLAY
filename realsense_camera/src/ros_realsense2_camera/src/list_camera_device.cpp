#include <librealsense2/rs.hpp>
#include <iostream>
#include <fstream>
#include <ros/ros.h>


void saveSerialToFile(std::vector<std::string> snum, std::string &file_path)
{
    if (!snum.empty())
    {
        // ifstream ifs;
        std::ofstream ofs;
        std::string str;

        ofs.open(file_path, std::ios::trunc);

        std::cout << std::endl;

        str = "#! /bin/bash\n";

        for (int i = 0; i < snum.size(); ++i)
        {
            std::string content = "export camera" + std::to_string(i) + "_serial_number=";
            std::string sn = snum[i];

            std::string ss = content + sn;

            if (i < snum.size() - 1)
            {
                str += ss + "\n";
            }
            else
            {
                str += ss;
            }
        }

        ofs << str << std::endl;
        ofs.close();
    }
}


std::vector<std::string> getRSSerialNum()
{
    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    rs2::device selected_device;
    std::vector<std::string> list_sn;
    if (devices.size() == 0)
    {
        std::cerr << "No device connected, please connect a RealSense device" << std::endl;
        ROS_ERROR("No device connected, please connect a RealSense device");

        //To help with the boilerplate code of waiting for a device to connect
        //The SDK provides the rs2::device_hub class
        rs2::device_hub device_hub(ctx);

        //Using the device_hub we can block the program until a device connects
        // selected_device = device_hub.wait_for_device();
        return list_sn;
    }
    else
    {
        std::cout << "Found the following devices:\n" << std::endl;

        // device_list is a "lazy" container of devices which allows
        //The device list provides 2 ways of iterating it
        //The first way is using an iterator (in this case hidden in the Range-based for loop)
//            int index = 0;
        for (rs2::device device: devices)
        {
            // std::cout << "  " << index++ << " : " << get_device_name(device) << std::endl;
            if (device.supports(RS2_CAMERA_INFO_SERIAL_NUMBER))
            {
                std::string sn = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
                list_sn.push_back(sn);
                std::cout << "Serial number: " << sn << std::endl;
            }
        }
    }

    return list_sn;
}

int main(int argc, char **argv)
try
{
    ros::init(argc, argv, "list_realsense_devices_node");

    std::string file_path = "/tmp/camera_config.sh";

    std::vector<std::string> list_sn;
    list_sn = getRSSerialNum();

    if (list_sn.empty())
    {
        std::cout << "list SN is empty" << std::endl;

        return -1;
    }

    // 写入文件
    saveSerialToFile(list_sn, file_path);

    return 0;
}
catch (const rs2::error &e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n"
              << e.what() << std::endl;

    return EXIT_FAILURE;
}