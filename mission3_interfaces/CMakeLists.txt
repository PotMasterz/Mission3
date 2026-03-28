cmake_minimum_required(VERSION 3.8)
project(mission3_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/AnalyzeRoom.srv"
  "srv/VerifyCompliance.srv"
  "srv/NavigateTo.srv"
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
