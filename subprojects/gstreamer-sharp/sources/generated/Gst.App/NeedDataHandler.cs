// This file was generated by the Gtk# code generator.
// Any changes made will be lost if regenerated.

namespace Gst.App {

	using System;

	public delegate void NeedDataHandler(object o, NeedDataArgs args);

	public class NeedDataArgs : GLib.SignalArgs {
		public uint Object{
			get {
				return (uint) Args [0];
			}
		}

	}
}