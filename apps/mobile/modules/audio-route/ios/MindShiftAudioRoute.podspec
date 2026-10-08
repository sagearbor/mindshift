Pod::Spec.new do |s|
  s.name           = 'MindShiftAudioRoute'
  s.version        = '1.0.0'
  s.summary        = 'Current audio output route + route-change events for Live Coach earpiece mode'
  s.description    = 'Local Expo module: AVAudioSession.currentRoute.outputs and routeChangeNotification.'
  s.author         = 'Sage Arbor'
  s.homepage       = 'https://github.com/sagearbor/mindshift'
  s.license        = { :type => 'Proprietary' }
  s.platforms      = { :ios => '16.4' }
  s.swift_version  = '5.9'
  s.source         = { git: '' }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'

  s.source_files = "**/*.{h,m,swift}"
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'SWIFT_COMPILATION_MODE' => 'wholemodule'
  }
end
