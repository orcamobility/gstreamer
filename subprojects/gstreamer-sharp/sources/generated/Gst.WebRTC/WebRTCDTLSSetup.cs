// This file was generated by the Gtk# code generator.
// Any changes made will be lost if regenerated.

namespace Gst.WebRTC {

	using System;
	using System.Runtime.InteropServices;

#region Autogenerated code
	[GLib.GType (typeof (Gst.WebRTC.WebRTCDTLSSetupGType))]
	public enum WebRTCDTLSSetup {

		None = 0,
		Actpass = 1,
		Active = 2,
		Passive = 3,
	}

	internal class WebRTCDTLSSetupGType {
		[DllImport ("gstwebrtc-1.0-0.dll", CallingConvention = CallingConvention.Cdecl)]
		static extern IntPtr gst_webrtc_dtls_setup_get_type ();

		public static GLib.GType GType {
			get {
				return new GLib.GType (gst_webrtc_dtls_setup_get_type ());
			}
		}
	}
#endregion
}