// This file was generated by the Gtk# code generator.
// Any changes made will be lost if regenerated.

namespace GES {

	using System;
	using System.Collections;
	using System.Collections.Generic;
	using System.Runtime.InteropServices;

#region Autogenerated code
	public partial class TrackElementAsset : GES.Asset {

		public TrackElementAsset (IntPtr raw) : base(raw) {}

		protected TrackElementAsset() : base(IntPtr.Zero)
		{
			CreateNativeObject (new string [0], new GLib.Value [0]);
		}

		[DllImport("ges-1.0", CallingConvention = CallingConvention.Cdecl)]
		static extern int ges_track_element_asset_get_track_type(IntPtr raw);

		[DllImport("ges-1.0", CallingConvention = CallingConvention.Cdecl)]
		static extern void ges_track_element_asset_set_track_type(IntPtr raw, int type);

		[GLib.Property ("track-type")]
		public GES.TrackType TrackType {
			get  {
				int raw_ret = ges_track_element_asset_get_track_type(Handle);
				GES.TrackType ret = (GES.TrackType) raw_ret;
				return ret;
			}
			set  {
				ges_track_element_asset_set_track_type(Handle, (int) value);
			}
		}

		static GetNaturalFramerateNativeDelegate GetNaturalFramerate_cb_delegate;
		static GetNaturalFramerateNativeDelegate GetNaturalFramerateVMCallback {
			get {
				if (GetNaturalFramerate_cb_delegate == null)
					GetNaturalFramerate_cb_delegate = new GetNaturalFramerateNativeDelegate (GetNaturalFramerate_cb);
				return GetNaturalFramerate_cb_delegate;
			}
		}

		static void OverrideGetNaturalFramerate (GLib.GType gtype)
		{
			OverrideGetNaturalFramerate (gtype, GetNaturalFramerateVMCallback);
		}

		static void OverrideGetNaturalFramerate (GLib.GType gtype, GetNaturalFramerateNativeDelegate callback)
		{
			unsafe {
				IntPtr* raw_ptr = (IntPtr*)(((long) gtype.GetClassPtr()) + (long) class_abi.GetFieldOffset("get_natural_framerate"));
				*raw_ptr = Marshal.GetFunctionPointerForDelegate((Delegate) callback);
			}
		}

		[UnmanagedFunctionPointer (CallingConvention.Cdecl)]
		delegate bool GetNaturalFramerateNativeDelegate (IntPtr inst, int framerate_n, int framerate_d);

		static bool GetNaturalFramerate_cb (IntPtr inst, int framerate_n, int framerate_d)
		{
			try {
				TrackElementAsset __obj = GLib.Object.GetObject (inst, false) as TrackElementAsset;
				bool __result;
				__result = __obj.OnGetNaturalFramerate (framerate_n, framerate_d);
				return __result;
			} catch (Exception e) {
				GLib.ExceptionManager.RaiseUnhandledException (e, true);
				// NOTREACHED: above call does not return.
				throw e;
			}
		}

		[GLib.DefaultSignalHandler(Type=typeof(GES.TrackElementAsset), ConnectionMethod="OverrideGetNaturalFramerate")]
		protected virtual bool OnGetNaturalFramerate (int framerate_n, int framerate_d)
		{
			return InternalGetNaturalFramerate (framerate_n, framerate_d);
		}

		private bool InternalGetNaturalFramerate (int framerate_n, int framerate_d)
		{
			GetNaturalFramerateNativeDelegate unmanaged = null;
			unsafe {
				IntPtr* raw_ptr = (IntPtr*)(((long) this.LookupGType().GetThresholdType().GetClassPtr()) + (long) class_abi.GetFieldOffset("get_natural_framerate"));
				unmanaged = (GetNaturalFramerateNativeDelegate) Marshal.GetDelegateForFunctionPointer(*raw_ptr, typeof(GetNaturalFramerateNativeDelegate));
			}
			if (unmanaged == null) return false;

			bool __result = unmanaged (this.Handle, framerate_n, framerate_d);
			return __result;
		}


		// Internal representation of the wrapped structure ABI.
		static GLib.AbiStruct _class_abi = null;
		static public new GLib.AbiStruct class_abi {
			get {
				if (_class_abi == null)
					_class_abi = new GLib.AbiStruct (new List<GLib.AbiField>{ 
						new GLib.AbiField("get_natural_framerate"
							, GES.Asset.class_abi.Fields
							, (uint) Marshal.SizeOf(typeof(IntPtr)) // get_natural_framerate
							, null
							, "_ges_reserved"
							, (uint) Marshal.SizeOf(typeof(IntPtr))
							, 0
							),
						new GLib.AbiField("_ges_reserved"
							, -1
							, (uint) Marshal.SizeOf(typeof(IntPtr)) * 3 // _ges_reserved
							, "get_natural_framerate"
							, null
							, (uint) Marshal.SizeOf(typeof(IntPtr))
							, 0
							),
					});

				return _class_abi;
			}
		}


		// End of the ABI representation.

		[DllImport("ges-1.0", CallingConvention = CallingConvention.Cdecl)]
		static extern IntPtr ges_track_element_asset_get_type();

		public static new GLib.GType GType { 
			get {
				IntPtr raw_ret = ges_track_element_asset_get_type();
				GLib.GType ret = new GLib.GType(raw_ret);
				return ret;
			}
		}

		[DllImport("ges-1.0", CallingConvention = CallingConvention.Cdecl)]
		static extern bool ges_track_element_asset_get_natural_framerate(IntPtr raw, int framerate_n, int framerate_d);

		public bool GetNaturalFramerate(int framerate_n, int framerate_d) {
			bool raw_ret = ges_track_element_asset_get_natural_framerate(Handle, framerate_n, framerate_d);
			bool ret = raw_ret;
			return ret;
		}


		static TrackElementAsset ()
		{
			GtkSharp.GstEditingServices.ObjectManager.Initialize ();
		}

		// Internal representation of the wrapped structure ABI.
		static GLib.AbiStruct _abi_info = null;
		static public new GLib.AbiStruct abi_info {
			get {
				if (_abi_info == null)
					_abi_info = new GLib.AbiStruct (new List<GLib.AbiField>{ 
						new GLib.AbiField("priv"
							, GES.Asset.abi_info.Fields
							, (uint) Marshal.SizeOf(typeof(IntPtr)) // priv
							, null
							, "__ges_reserved"
							, (uint) Marshal.SizeOf(typeof(IntPtr))
							, 0
							),
						new GLib.AbiField("__ges_reserved"
							, -1
							, (uint) Marshal.SizeOf(typeof(IntPtr)) * 4 // __ges_reserved
							, "priv"
							, null
							, (uint) Marshal.SizeOf(typeof(IntPtr))
							, 0
							),
					});

				return _abi_info;
			}
		}


		// End of the ABI representation.

#endregion
	}
}